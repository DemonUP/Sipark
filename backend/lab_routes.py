import asyncio
import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

import cv2
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from ultralytics import YOLO

import detection

logger = logging.getLogger("sipark.lab")

router = APIRouter(tags=["lab"])

LAB_CONF = 0.25

# `min_area` se conserva solo por compatibilidad de la API y de la cache: el
# frontend lo envia en /lab/annotated y /lab/report. Ya no filtra nada. El
# nucleo descarta duplicados, fragmentos y cajas incoherentes con la escala que
# midio en la propia imagen, de modo que un area minima fija sobraba y excluia
# motos pequenas, lejanas u ocluidas.
LAB_MIN_BBOX_AREA = 5000

BASE_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = BASE_DIR / "test_images"

# Modelo EXCLUSIVO para lab — separado del modelo principal de /api/ingest
# Esto evita conflictos de estado interno de PyTorch entre llamadas concurrentes
_LAB_MODEL_PATH = str(BASE_DIR / os.getenv("SIPARK_MODEL", "yolo11m.pt"))
_lab_model: YOLO | None = None


def _get_lab_model() -> YOLO:
    """Carga el modelo la primera vez (lazy) y lo reutiliza. Siempre en el executor thread."""
    global _lab_model
    if _lab_model is None:
        logger.info("Cargando modelo lab desde %s", _LAB_MODEL_PATH)
        _lab_model = YOLO(_LAB_MODEL_PATH)
        logger.info("Modelo lab listo")
    return _lab_model


# Un solo worker: garantiza que YOLO corra siempre en el mismo thread (thread-safe)
_analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lab_yolo")
_analysis_cache = {"key": None, "value": None}
_analysis_state_lock = asyncio.Lock()
_analysis_inflight: dict[tuple, asyncio.Task] = {}


def _lighting_class(mean_brightness: float) -> str:
    if mean_brightness < 80:
        return "Brillo bajo"
    if mean_brightness <= 160:
        return "Brillo medio"
    return "Brillo alto"


def _angle_class(aspect_ratio: float) -> str:
    if aspect_ratio > 1.3:
        return "Horizontal"
    if aspect_ratio < 0.7:
        return "Vertical"
    return "Compacta"


def _resolve_image_path(filename: str) -> Path:
    decoded = unquote(filename)
    candidate = (TEST_IMAGES_DIR / decoded).resolve()
    if candidate.parent != TEST_IMAGES_DIR.resolve() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Imagen no encontrada")
    return candidate


def _list_image_files() -> list[str]:
    allowed = {".jpg", ".jpeg", ".png"}
    return sorted(
        [path.name for path in TEST_IMAGES_DIR.iterdir() if path.is_file() and path.suffix.lower() in allowed]
    )


def _find_motorcycle_class_id(model) -> int | None:
    for class_id, name in model.names.items():
        if str(name).lower() == "motorcycle":
            return int(class_id)
    return None


def _lab_params(conf: float) -> detection.DetectParams:
    """Parametros del nucleo para el laboratorio.

    Se parte de los valores por defecto del nucleo y se leen las mismas
    variables de entorno que usa produccion, para que el laboratorio mida el
    mismo pipeline que corre el monitoreo. La unica diferencia deliberada es la
    confianza, que llega por la API y se usa tal cual, sin elevarla.
    """
    return detection.DetectParams(
        conf=float(conf),
        truncated_conf=float(os.getenv("SIPARK_TRUNCATED_CONF", "0.45")),
        iou=float(os.getenv("SIPARK_IOU", "0.60")),
        probe_imgsz=int(os.getenv("SIPARK_PROBE_IMGSZ", "960")),
        tile_imgsz=int(os.getenv("SIPARK_TILE_IMGSZ", "640")),
        tile_overlap=float(os.getenv("SIPARK_OVERLAP", "0.35")),
        merge_iou=float(os.getenv("SIPARK_MERGE_IOU", "0.55")),
        max_tiles=int(os.getenv("SIPARK_MAX_TILES", "60")),
        preprocess=os.getenv("SIPARK_PREPROCESS", "0") == "1",
        class_names=("motorcycle",),
    )


def _confidence_band(score: float) -> str:
    if score >= 0.8:
        return "Alta (>=0.80)"
    if score >= 0.6:
        return "Media (0.60-0.79)"
    return "Baja (<0.60)"


def _analyze_dataset_sync(model, conf: float, min_area: int) -> dict:
    started = time.perf_counter()
    image_files = _list_image_files()
    motorcycle_class_id = _find_motorcycle_class_id(model)
    if motorcycle_class_id is None:
        raise HTTPException(status_code=422, detail="El modelo no contiene la clase motorcycle")

    params = _lab_params(conf)

    images: list[dict] = []
    angle_distribution = {"Compacta": 0, "Horizontal": 0, "Vertical": 0}
    lighting_distribution = {"Brillo alto": 0, "Brillo medio": 0, "Brillo bajo": 0}
    viewpoint_distribution = {"superior": 0, "oblicua": 0, "indeterminado": 0}
    skipped_images = []
    rejected_totals = Counter()
    confidence_bands = {"Alta (>=0.80)": 0, "Media (0.60-0.79)": 0, "Baja (<0.60)": 0}
    total_detections = 0
    total_tiles = 0
    total_inference_calls = 0
    confidence_values: list[float] = []
    bbox_areas: list[float] = []
    images_with_detections = 0

    for filename in image_files:
        image_path = TEST_IMAGES_DIR / filename
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            skipped_images.append(filename)
            continue

        image_h, image_w = img_bgr.shape[:2]
        image_area = float(image_w * image_h)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())
        lighting_class = _lighting_class(mean_brightness)
        lighting_distribution[lighting_class] += 1

        # Nucleo comun: la misma inferencia que corre el monitoreo. Antes el
        # laboratorio predecia por su cuenta a 640 sobre la imagen completa y
        # aplicaba filtros de lado, area y proporcion propios, asi que sus
        # numeros no validaban produccion.
        res = detection.detect(model, img_bgr, params)
        scene = res.scene

        # `res.dropped` incluye las cortadas que si se aceptaron, que no son un
        # descarte: se separan para no contarlas como rechazo.
        #
        # Compromiso aceptado: los motivos ya no son los del laboratorio
        # (min_side, min_area, max_area, aspect_ratio) sino los de la fusion del
        # nucleo. El frontend traduce los motivos que conoce y muestra el resto
        # tal cual (`FILTER_LABELS[reason] ?? reason` en LabPage.jsx), asi que no
        # se rompe; las etiquetas nuevas se pueden anadir alli cuando convenga.
        rejected = Counter({
            reason: int(count)
            for reason, count in res.dropped.items()
            if reason != "cortadas_aceptadas" and count
        })
        accepted_truncated = int(res.dropped.get("cortadas_aceptadas", 0))
        candidate_count = int(res.candidates)
        total_tiles += int(res.tiles_used)
        total_inference_calls += int(res.inference_calls)

        kept_detections: list[dict] = []
        for det in res.detections:
            x1, y1, x2, y2 = (float(v) for v in det.box)
            width = max(0.0, x2 - x1)
            height = max(1.0, y2 - y1)
            bbox_area = float(width * height)
            aspect_ratio = float(width / height)
            kept_detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "bbox_area_px": bbox_area,
                    "confidence": float(det.score),
                    "class_name": str(det.cls_name or model.names.get(int(det.cls), det.cls)),
                    "aspect_ratio": aspect_ratio,
                    # Clave historica: describe la forma de la caja, no el
                    # angulo real de la moto.
                    "angle_class": _angle_class(aspect_ratio),
                    "truncated": bool(det.truncated),
                    "source": str(det.source),
                }
            )

        for idx, det_item in enumerate(kept_detections):
            det_item["detection_idx"] = idx
            det_item["bbox"] = [round(v, 2) for v in det_item["bbox"]]
            det_item["bbox_area_px"] = round(det_item["bbox_area_px"], 2)
            det_item["confidence"] = round(det_item["confidence"], 4)
            det_item["aspect_ratio"] = round(det_item["aspect_ratio"], 4)
            angle_distribution[det_item["angle_class"]] += 1
            confidence_values.append(det_item["confidence"])
            bbox_areas.append(det_item["bbox_area_px"])
            confidence_bands[_confidence_band(det_item["confidence"])] += 1

        detection_count = len(kept_detections)
        rejected_totals.update(rejected)
        viewpoint_distribution[scene.viewpoint] = viewpoint_distribution.get(scene.viewpoint, 0) + 1
        bbox_coverage_pct = round(
            (sum(det["bbox_area_px"] for det in kept_detections) / image_area) * 100, 2
        ) if image_area else 0.0

        if detection_count > 0:
            images_with_detections += 1
            total_detections += detection_count

        predominant_angle = "--"
        if kept_detections:
            angle_counts = {}
            for det in kept_detections:
                angle_counts[det["angle_class"]] = angle_counts.get(det["angle_class"], 0) + 1
            predominant_angle = max(angle_counts.items(), key=lambda item: item[1])[0]

        images.append(
            {
                "filename": filename,
                "candidate_count": candidate_count,
                "rejected_by_reason": dict(rejected),
                # El nucleo limita a `max_det` cajas por llamada al detector.
                # Si el total de candidatos no llega a ese tope, ninguna llamada
                # pudo saturarse; si lo alcanza, alguna pudo truncar su salida.
                "inference_limit_reached": candidate_count >= int(params.max_det),
                # Geometria medida en esta foto: dice como se adapto el sistema.
                "viewpoint": scene.viewpoint,
                "relative_slope": round(float(scene.relative_slope), 4),
                "median_side_px": round(float(scene.median_side), 2),
                "scene_reliable": bool(scene.reliable),
                "probe_count": int(scene.probe_count),
                "tiles_used": int(res.tiles_used),
                "inference_calls": int(res.inference_calls),
                "truncated_candidates": int(res.truncated_candidates),
                "accepted_truncated": accepted_truncated,
                "image_w": int(image_w),
                "image_h": int(image_h),
                "mean_brightness": round(mean_brightness, 2),
                "lighting_class": lighting_class,
                "predominant_angle": predominant_angle,
                "detection_count": detection_count,
                "avg_confidence": round(
                    float(sum(d["confidence"] for d in kept_detections) / detection_count), 4
                )
                if detection_count
                else 0.0,
                "bbox_coverage_pct": bbox_coverage_pct,
                "detections": kept_detections,
            }
        )

    total_images = len(images)
    images_without_detections = total_images - images_with_detections
    avg_confidence = round(float(np.mean(confidence_values)), 4) if confidence_values else 0.0
    median_confidence = round(float(np.median(confidence_values)), 4) if confidence_values else 0.0
    avg_bbox_area = round(float(np.mean(bbox_areas)), 2) if bbox_areas else 0.0
    avg_bbox_coverage = round(float(np.mean([img["bbox_coverage_pct"] for img in images])), 2) if images else 0.0
    high_conf_share = round(confidence_bands["Alta (>=0.80)"] / total_detections, 4) if total_detections else 0.0

    analysis = {
        "ok": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "params": {
            "conf": round(float(conf), 4),
            # Se conserva por compatibilidad de la API y de la cache. Ya no
            # filtra: no hay area minima en el nucleo comun.
            "min_area": int(min_area),
            "min_area_aplica": False,
            "model": _LAB_MODEL_PATH,
            "nucleo": "detection.detect (compartido con el monitoreo)",
            "probe_imgsz": int(params.probe_imgsz),
            "probe_conf": round(float(params.probe_conf), 4),
            "tile_imgsz": int(params.tile_imgsz),
            "tile_overlap": round(float(params.tile_overlap), 4),
            "target_obj_frac": round(float(params.target_obj_frac), 4),
            "bands": int(params.bands),
            "scales": int(params.scales),
            "max_tiles": int(params.max_tiles),
            "iou": round(float(params.iou), 4),
            "merge_iou": round(float(params.merge_iou), 4),
            "contain_thr": round(float(params.contain_thr), 4),
            "scale_tol": round(float(params.scale_tol), 4),
            "truncated_conf": round(float(params.truncated_conf), 4),
            "max_det": int(params.max_det),
            "preprocess": bool(params.preprocess),
            "class_names": ", ".join(params.class_names),
        },
        "methodology": {
            "version": 3,
            "notes": [
                "La deteccion usa el nucleo comun detection.detect, el mismo del monitoreo: un resultado del laboratorio si describe el pipeline real.",
                "El nucleo mide la geometria en cada imagen (vista superior u oblicua) y elige por franjas el tamano de mosaico; la vista y los mosaicos usados se reportan por imagen.",
                "Se eliminaron los filtros de lado minimo, area minima, area maxima y proporcion: excluian motos pequenas, lejanas u ocluidas. El parametro min_area se conserva en la API pero no filtra.",
                "La confianza solicitada se usa tal cual. El sondeo inicial de escena corre con su propia confianza (probe_conf) porque sirve para medir la escala, no para aceptar cajas.",
                "La confianza del modelo no mide precision ni recall; se requieren anotaciones reales para evaluarlos.",
                "La tasa de deteccion es la fraccion de imagenes con alguna deteccion aceptada.",
                "El brillo describe pixeles, no la hora del dia. La forma de la caja no determina el angulo de la moto.",
                "Los descartes provienen de la fusion del nucleo: duplicados entre mosaicos (nms), cajas contenidas en otra (contenida), tamano incoherente con la escala medida (escala) y vistas parciales con poca confianza (cortada_debil).",
                "Los candidatos son las cajas de todos los mosaicos antes de fusionar, asi que una misma moto aporta varios candidatos.",
                "La cobertura suma areas de cajas; puede superar 100% cuando se solapan.",
            ],
        },
        "audit": {
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "files_found": len(image_files),
            "skipped_images": skipped_images,
            "candidates": sum(img["candidate_count"] for img in images),
            "rejected_by_reason": dict(rejected_totals),
            "tiles_used": total_tiles,
            "inference_calls": total_inference_calls,
            "truncated_candidates": sum(img["truncated_candidates"] for img in images),
            "accepted_truncated": sum(img["accepted_truncated"] for img in images),
        },
        "total_images": total_images,
        "total_detections": total_detections,
        "images": images,
        "kpis": {
            "avg_confidence": avg_confidence,
            "median_confidence": median_confidence,
            "detection_rate": round(images_with_detections / total_images, 4) if total_images else 0.0,
            "images_with_detections": images_with_detections,
            "images_without_detections": images_without_detections,
            "avg_detections_per_positive_image": round(total_detections / images_with_detections, 2)
            if images_with_detections
            else 0.0,
            "avg_bbox_area_px": avg_bbox_area,
            "avg_bbox_coverage_pct": avg_bbox_coverage,
            "high_confidence_share": high_conf_share,
            "angle_distribution": angle_distribution,
            "lighting_distribution": lighting_distribution,
            "confidence_bands": confidence_bands,
            "viewpoint_distribution": viewpoint_distribution,
            "avg_tiles_per_image": round(total_tiles / total_images, 2) if total_images else 0.0,
        },
    }
    return analysis


def _dataset_signature() -> tuple:
    return tuple(
        (name, stat.st_size, stat.st_mtime_ns)
        for name in _list_image_files()
        for stat in [(TEST_IMAGES_DIR / name).stat()]
    )


async def _get_or_run_analysis(request: Request, conf: float, min_area: int) -> dict:
    signature = await asyncio.to_thread(_dataset_signature)
    cache_key = (float(conf), int(min_area), signature)

    async def run_and_cache() -> dict:
        try:
            def run_job() -> dict:
                return _analyze_dataset_sync(_get_lab_model(), conf, min_area)

            analysis = await asyncio.get_running_loop().run_in_executor(_analysis_executor, run_job)
            # Do not publish a result if the dataset changed during inference.
            if await asyncio.to_thread(_dataset_signature) != signature:
                raise HTTPException(status_code=409, detail="Las imagenes cambiaron durante el analisis. Ejecutelo de nuevo.")
            async with _analysis_state_lock:
                _analysis_cache.update(key=cache_key, value=analysis)
            return analysis
        finally:
            async with _analysis_state_lock:
                _analysis_inflight.pop(cache_key, None)

    async with _analysis_state_lock:
        if _analysis_cache["key"] == cache_key and _analysis_cache["value"] is not None:
            return _analysis_cache["value"]
        task = _analysis_inflight.get(cache_key)
        if task is None:
            task = asyncio.create_task(run_and_cache())
            # Retrieve failures even when every HTTP caller has disconnected.
            task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            _analysis_inflight[cache_key] = task
    return await asyncio.shield(task)


def _draw_detection_overlay(image_path: Path, image_analysis: dict) -> bytes:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=404, detail="Imagen no encontrada")

    for det in image_analysis.get("detections", []):
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        confidence = det["confidence"]
        color = (34, 197, 94) if confidence >= 0.75 else (14, 165, 233) if confidence >= 0.6 else (245, 158, 11)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label = f"moto {confidence:.2f} {det['angle_class']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        label_y = max(0, y1 - th - 10)
        cv2.rectangle(image, (x1, label_y), (x1 + tw + 10, label_y + th + 10), color, -1)
        cv2.putText(
            image,
            label,
            (x1 + 5, label_y + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo generar overlay")
    return encoded.tobytes()


def _build_report_bytes(analysis: dict) -> bytes:
    output = BytesIO()
    kpis = analysis["kpis"]
    detail_rows = []
    image_rows = []

    for image in analysis["images"]:
        image_rows.append(
            {
                "Archivo": image["filename"],
                "Detecciones": image["detection_count"],
                "Confianza Promedio": image["avg_confidence"],
                "Cobertura (%)": image["bbox_coverage_pct"],
                "Brillo Medio": image["mean_brightness"],
                "Iluminacion": image["lighting_class"],
                "Forma Predominante": image["predominant_angle"],
                "Resolucion": f"{image['image_w']} x {image['image_h']}",
                "Tiene Deteccion": "Si" if image["detection_count"] > 0 else "No",
                # Geometria medida por el nucleo en esta imagen.
                "Vista": image.get("viewpoint", "indeterminado"),
                "Pendiente Relativa": image.get("relative_slope", 0.0),
                "Lado Mediano (px)": image.get("median_side_px", 0.0),
                "Mosaicos": image.get("tiles_used", 0),
            }
        )
        for detection in image["detections"]:
            x1, y1, x2, y2 = detection["bbox"]
            detail_rows.append(
                {
                    "Archivo": image["filename"],
                    "Deteccion #": detection["detection_idx"] + 1,
                    "X1": x1,
                    "Y1": y1,
                    "X2": x2,
                    "Y2": y2,
                    "Area (px)": detection["bbox_area_px"],
                    "Confianza": detection["confidence"],
                    "Clase": detection["class_name"],
                    "Aspect Ratio": detection["aspect_ratio"],
                    "Forma de Caja": detection["angle_class"],
                    "Brillo Medio": image["mean_brightness"],
                    "Iluminacion": image["lighting_class"],
                    "Cobertura Imagen (%)": image["bbox_coverage_pct"],
                    "Vista Parcial": "Si" if detection.get("truncated") else "No",
                    "Vista": image.get("viewpoint", "indeterminado"),
                }
            )

    image_df = pd.DataFrame(image_rows, columns=[
        "Archivo", "Detecciones", "Confianza Promedio", "Cobertura (%)", "Brillo Medio",
        "Iluminacion", "Forma Predominante", "Resolucion", "Tiene Deteccion",
        "Vista", "Pendiente Relativa", "Lado Mediano (px)", "Mosaicos",
    ])
    detail_df = pd.DataFrame(
        detail_rows,
        columns=[
            "Archivo",
            "Deteccion #",
            "X1",
            "Y1",
            "X2",
            "Y2",
            "Area (px)",
            "Confianza",
            "Clase",
            "Aspect Ratio",
            "Forma de Caja",
            "Brillo Medio",
            "Iluminacion",
            "Cobertura Imagen (%)",
            "Vista Parcial",
            "Vista",
        ],
    )

    if not image_df.empty:
        summary_sorted = image_df.sort_values(
            by=["Detecciones", "Confianza Promedio", "Cobertura (%)"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        positive_df = image_df[image_df["Detecciones"] > 0].copy()
        top_images_df = summary_sorted.head(10).copy()
        bottom_images_df = image_df.sort_values(
            by=["Detecciones", "Confianza Promedio", "Cobertura (%)"],
            ascending=[True, True, True],
        ).head(10).reset_index(drop=True)
        lighting_summary_df = (
            image_df.groupby("Iluminacion", dropna=False)
            .agg(
                Imagenes=("Archivo", "count"),
                ImagenesConDet=("Detecciones", lambda s: int((s > 0).sum())),
                Detecciones=("Detecciones", "sum"),
                ConfianzaMedia=("Confianza Promedio", "mean"),
                CoberturaMediaPct=("Cobertura (%)", "mean"),
                BrilloMedio=("Brillo Medio", "mean"),
            )
            .reset_index()
        )
        angle_summary_df = (
            image_df.groupby("Forma Predominante", dropna=False)
            .agg(
                Imagenes=("Archivo", "count"),
                Detecciones=("Detecciones", "sum"),
                ConfianzaMedia=("Confianza Promedio", "mean"),
                CoberturaMediaPct=("Cobertura (%)", "mean"),
            )
            .reset_index()
        )
    else:
        summary_sorted = pd.DataFrame(columns=image_df.columns)
        positive_df = pd.DataFrame(columns=image_df.columns)
        top_images_df = pd.DataFrame(columns=image_df.columns)
        bottom_images_df = pd.DataFrame(columns=image_df.columns)
        lighting_summary_df = pd.DataFrame(
            columns=["Iluminacion", "Imagenes", "ImagenesConDet", "Detecciones", "ConfianzaMedia", "CoberturaMediaPct", "BrilloMedio"]
        )
        angle_summary_df = pd.DataFrame(
            columns=["Forma Predominante", "Imagenes", "Detecciones", "ConfianzaMedia", "CoberturaMediaPct"]
        )

    if not detail_df.empty:
        confidence_summary_df = (
            detail_df.assign(
                BandaConfianza=detail_df["Confianza"].apply(
                    lambda v: "Alta (>=0.80)" if v >= 0.8 else "Media (0.60-0.79)" if v >= 0.6 else "Baja (<0.60)"
                )
            )
            .groupby("BandaConfianza", dropna=False)
            .agg(
                Detecciones=("Archivo", "count"),
                ConfianzaMedia=("Confianza", "mean"),
                AreaMediaPx=("Area (px)", "mean"),
            )
            .reset_index()
        )
        detail_sorted = detail_df.sort_values(by=["Confianza", "Area (px)"], ascending=[False, False]).reset_index(drop=True)
        top_detections_df = detail_sorted.head(25).copy()
    else:
        confidence_summary_df = pd.DataFrame(columns=["BandaConfianza", "Detecciones", "ConfianzaMedia", "AreaMediaPx"])
        top_detections_df = pd.DataFrame(columns=detail_df.columns)

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame({"Notas de interpretacion": analysis["methodology"]["notes"]}).to_excel(
            writer, sheet_name="Metodologia", index=False
        )
        writer.sheets["Metodologia"].set_column("A:A", 110)
        audit_rows = [
            {"Archivo": item["filename"], "Candidatos (mosaicos)": item["candidate_count"],
             "Aceptadas": item["detection_count"], "Limite del detector alcanzado": item["inference_limit_reached"],
             # Geometria con la que el nucleo se adapto a esta imagen.
             "Vista": item.get("viewpoint", "indeterminado"),
             "Pendiente Relativa": item.get("relative_slope", 0.0),
             "Lado Mediano (px)": item.get("median_side_px", 0.0),
             "Geometria Fiable": item.get("scene_reliable", False),
             "Mosaicos": item.get("tiles_used", 0),
             "Llamadas Inferencia": item.get("inference_calls", 0),
             "Cortadas Candidatas": item.get("truncated_candidates", 0),
             "Cortadas Aceptadas": item.get("accepted_truncated", 0),
             **item["rejected_by_reason"]}
            for item in analysis["images"]
        ]
        pd.DataFrame(audit_rows).to_excel(writer, sheet_name="Auditoria Filtros", index=False)
        pd.DataFrame({"Archivo ilegible": analysis["audit"]["skipped_images"]}).to_excel(
            writer, sheet_name="Imagenes Omitidas", index=False
        )
        workbook = writer.book

        title_fmt = workbook.add_format({"bold": True, "font_size": 20, "font_color": "#0F172A"})
        subtitle_fmt = workbook.add_format({"font_size": 10, "font_color": "#475569"})
        section_fmt = workbook.add_format({"bold": True, "bg_color": "#DBEAFE", "border": 1, "font_color": "#1E3A8A"})
        label_fmt = workbook.add_format({"bold": True, "font_color": "#0F172A"})
        value_fmt = workbook.add_format({"num_format": "0.00"})
        percent_fmt = workbook.add_format({"num_format": "0.00%"})
        integer_fmt = workbook.add_format({"num_format": "0"})
        metric_card_fmt = workbook.add_format(
            {"bold": True, "font_size": 16, "align": "center", "valign": "vcenter", "bg_color": "#F8FAFC", "border": 1}
        )
        metric_label_fmt = workbook.add_format(
            {"font_size": 9, "align": "center", "valign": "vcenter", "bg_color": "#E2E8F0", "border": 1, "font_color": "#334155"}
        )
        note_fmt = workbook.add_format({"text_wrap": True, "valign": "top", "font_color": "#475569"})
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#E2E8F0", "border": 1, "font_color": "#0F172A"})

        dashboard = workbook.add_worksheet("Dashboard")
        writer.sheets["Dashboard"] = dashboard
        dashboard.set_column("A:A", 28)
        dashboard.set_column("B:B", 14)
        dashboard.set_column("C:C", 20)
        dashboard.set_column("D:D", 14)
        dashboard.set_column("F:I", 18)

        dashboard.write("A1", "Sipark -- Reporte de Laboratorio", title_fmt)
        dashboard.write("A2", f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_fmt)
        dashboard.write(
            "A3",
            (
                f"Parametros: conf={analysis['params']['conf']} | "
                f"min_area={analysis['params']['min_area']} (no filtra) | "
                f"sondeo={analysis['params']['probe_imgsz']} | mosaico={analysis['params']['tile_imgsz']} | "
                f"solape={analysis['params']['tile_overlap']} | fusion_iou={analysis['params']['merge_iou']} | "
                f"max_mosaicos={analysis['params']['max_tiles']}"
            ),
            subtitle_fmt,
        )

        cards = [
            ("A5:B5", "Imagenes", analysis["total_images"]),
            ("C5:D5", "Detecciones", analysis["total_detections"]),
            ("A7:B7", "Tasa de deteccion", kpis["detection_rate"]),
            ("C7:D7", "Confianza promedio", kpis["avg_confidence"]),
            ("A9:B9", "Confianza mediana", kpis["median_confidence"]),
            ("C9:D9", "Cobertura promedio", kpis["avg_bbox_coverage_pct"] / 100.0),
        ]
        for rng, label, value in cards:
            left, right = rng.split(":")
            dashboard.merge_range(rng, "", metric_card_fmt)
            dashboard.write(left, value, percent_fmt if label in {"Tasa de deteccion", "Cobertura promedio"} else value_fmt if isinstance(value, float) else integer_fmt)
            dashboard.write(right, label, metric_label_fmt)

        dashboard.write("A12", "Resumen Ejecutivo", section_fmt)
        executive_note = (
            f"Se procesaron {analysis['total_images']} imagenes close-up con {analysis['total_detections']} detecciones "
            f"aceptadas por los filtros. {kpis['images_with_detections']} imagenes presentaron al menos una deteccion y "
            f"{kpis['images_without_detections']} no presentaron detecciones aceptadas. La confianza media fue "
            f"{kpis['avg_confidence']:.2f}, la mediana {kpis['median_confidence']:.2f} y la cobertura media "
            f"de bounding boxes fue {kpis['avg_bbox_coverage_pct']:.2f}%."
        )
        dashboard.merge_range("A13:D16", executive_note, note_fmt)

        dashboard.write("A18", "Indicadores de Calidad", section_fmt)
        quality_rows = [
            ("Imagenes con deteccion", kpis["images_with_detections"], integer_fmt),
            ("Imagenes sin deteccion", kpis["images_without_detections"], integer_fmt),
            ("Detecciones / imagen positiva", kpis["avg_detections_per_positive_image"], value_fmt),
            ("Area promedio bbox (px)", kpis["avg_bbox_area_px"], value_fmt),
            ("Share alta confianza", kpis["high_confidence_share"], percent_fmt),
        ]
        for row_idx, (label, value, fmt) in enumerate(quality_rows, start=19):
            dashboard.write(f"A{row_idx}", label, label_fmt)
            dashboard.write(f"B{row_idx}", value, fmt)

        dashboard.write("D18", "Bandas de Confianza", section_fmt)
        confidence_labels = ["Alta (>=0.80)", "Media (0.60-0.79)", "Baja (<0.60)"]
        for row_idx, label in enumerate(confidence_labels, start=19):
            dashboard.write(f"D{row_idx}", label, label_fmt)
            dashboard.write(f"E{row_idx}", kpis["confidence_bands"][label], integer_fmt)

        dashboard.write("A26", "Distribucion por Forma", section_fmt)
        angle_labels = ["Compacta", "Horizontal", "Vertical"]
        for row_idx, label in enumerate(angle_labels, start=27):
            dashboard.write(f"A{row_idx}", label, label_fmt)
            dashboard.write(f"B{row_idx}", kpis["angle_distribution"][label], integer_fmt)

        dashboard.write("D26", "Distribucion por Iluminacion", section_fmt)
        lighting_labels = ["Brillo alto", "Brillo medio", "Brillo bajo"]
        for row_idx, label in enumerate(lighting_labels, start=27):
            dashboard.write(f"D{row_idx}", label, label_fmt)
            dashboard.write(f"E{row_idx}", kpis["lighting_distribution"][label], integer_fmt)

        # Geometria medida: cuantas imagenes resultaron vista superior, oblicua o
        # no medible, y cuantos mosaicos costo en promedio.
        dashboard.write("A31", "Geometria Medida", section_fmt)
        viewpoint_labels = ["superior", "oblicua", "indeterminado"]
        for row_idx, label in enumerate(viewpoint_labels, start=32):
            dashboard.write(f"A{row_idx}", label, label_fmt)
            dashboard.write(f"B{row_idx}", kpis.get("viewpoint_distribution", {}).get(label, 0), integer_fmt)
        dashboard.write("A35", "Mosaicos por imagen", label_fmt)
        dashboard.write("B35", kpis.get("avg_tiles_per_image", 0.0), value_fmt)

        angle_chart = workbook.add_chart({"type": "column"})
        angle_chart.add_series(
            {
                "name": "Forma",
                "categories": "=Dashboard!$A$27:$A$29",
                "values": "=Dashboard!$B$27:$B$29",
                "fill": {"color": "#2563EB"},
                "border": {"none": True},
            }
        )
        angle_chart.set_title({"name": "Distribucion por forma"})
        angle_chart.set_legend({"none": True})
        angle_chart.set_size({"width": 420, "height": 240})
        dashboard.insert_chart("G5", angle_chart)

        lighting_chart = workbook.add_chart({"type": "pie"})
        lighting_chart.add_series(
            {
                "name": "Iluminacion",
                "categories": "=Dashboard!$D$27:$D$29",
                "values": "=Dashboard!$E$27:$E$29",
                "data_labels": {"percentage": True},
            }
        )
        lighting_chart.set_title({"name": "Distribucion por iluminacion"})
        lighting_chart.set_size({"width": 380, "height": 240})
        dashboard.insert_chart("G19", lighting_chart)

        if not summary_sorted.empty:
            top_n = min(len(summary_sorted), 12)
            rank_chart = workbook.add_chart({"type": "bar"})
            rank_chart.add_series(
                {
                    "name": "Detecciones",
                    "categories": f"='Resumen por Imagen'!$A$2:$A${top_n + 1}",
                    "values": f"='Resumen por Imagen'!$B$2:$B${top_n + 1}",
                    "fill": {"color": "#10B981"},
                    "border": {"none": True},
                }
            )
            rank_chart.set_title({"name": "Top imagenes por detecciones"})
            rank_chart.set_legend({"none": True})
            rank_chart.set_size({"width": 520, "height": 340})
            dashboard.insert_chart("G33", rank_chart)

        params_df = pd.DataFrame(
            [
                {"Parametro": key, "Valor": value}
                for key, value in analysis["params"].items()
            ]
        )
        params_df.to_excel(writer, sheet_name="Parametros", index=False)
        params_sheet = writer.sheets["Parametros"]
        params_sheet.freeze_panes(1, 0)
        params_sheet.set_column("A:A", 28)
        params_sheet.set_column("B:B", 18)
        for col_idx, column in enumerate(params_df.columns):
            params_sheet.write(0, col_idx, column, header_fmt)

        summary_sorted.to_excel(writer, sheet_name="Resumen por Imagen", index=False)
        summary_sheet = writer.sheets["Resumen por Imagen"]
        summary_sheet.freeze_panes(1, 0)
        for col_idx, column in enumerate(summary_sorted.columns):
            summary_sheet.write(0, col_idx, column, header_fmt)
            values = [column] + summary_sorted[column].astype(str).tolist() if not summary_sorted.empty else [column]
            summary_sheet.set_column(col_idx, col_idx, min(max(len(v) for v in values) + 2, 42))
        summary_sheet.autofilter(0, 0, max(len(summary_sorted), 1), len(summary_sorted.columns) - 1)

        top_images_df.to_excel(writer, sheet_name="Top Imagenes", index=False)
        top_sheet = writer.sheets["Top Imagenes"]
        top_sheet.freeze_panes(1, 0)
        for col_idx, column in enumerate(top_images_df.columns):
            top_sheet.write(0, col_idx, column, header_fmt)
            values = [column] + top_images_df[column].astype(str).tolist() if not top_images_df.empty else [column]
            top_sheet.set_column(col_idx, col_idx, min(max(len(v) for v in values) + 2, 42))

        bottom_images_df.to_excel(writer, sheet_name="Bottom Imagenes", index=False)
        bottom_sheet = writer.sheets["Bottom Imagenes"]
        bottom_sheet.freeze_panes(1, 0)
        for col_idx, column in enumerate(bottom_images_df.columns):
            bottom_sheet.write(0, col_idx, column, header_fmt)
            values = [column] + bottom_images_df[column].astype(str).tolist() if not bottom_images_df.empty else [column]
            bottom_sheet.set_column(col_idx, col_idx, min(max(len(v) for v in values) + 2, 42))

        lighting_summary_df.to_excel(writer, sheet_name="Analisis Iluminacion", index=False)
        lighting_sheet = writer.sheets["Analisis Iluminacion"]
        lighting_sheet.freeze_panes(1, 0)
        for col_idx, column in enumerate(lighting_summary_df.columns):
            lighting_sheet.write(0, col_idx, column, header_fmt)
            values = [column] + lighting_summary_df[column].astype(str).tolist() if not lighting_summary_df.empty else [column]
            lighting_sheet.set_column(col_idx, col_idx, min(max(len(v) for v in values) + 2, 28))

        angle_summary_df.to_excel(writer, sheet_name="Analisis Forma", index=False)
        angle_sheet = writer.sheets["Analisis Forma"]
        angle_sheet.freeze_panes(1, 0)
        for col_idx, column in enumerate(angle_summary_df.columns):
            angle_sheet.write(0, col_idx, column, header_fmt)
            values = [column] + angle_summary_df[column].astype(str).tolist() if not angle_summary_df.empty else [column]
            angle_sheet.set_column(col_idx, col_idx, min(max(len(v) for v in values) + 2, 28))

        confidence_summary_df.to_excel(writer, sheet_name="Calidad Detecciones", index=False)
        conf_sheet = writer.sheets["Calidad Detecciones"]
        conf_sheet.freeze_panes(1, 0)
        for col_idx, column in enumerate(confidence_summary_df.columns):
            conf_sheet.write(0, col_idx, column, header_fmt)
            values = [column] + confidence_summary_df[column].astype(str).tolist() if not confidence_summary_df.empty else [column]
            conf_sheet.set_column(col_idx, col_idx, min(max(len(v) for v in values) + 2, 28))

        top_detections_df.to_excel(writer, sheet_name="Top Detecciones", index=False)
        top_det_sheet = writer.sheets["Top Detecciones"]
        top_det_sheet.freeze_panes(1, 0)
        for col_idx, column in enumerate(top_detections_df.columns):
            top_det_sheet.write(0, col_idx, column, header_fmt)
            values = [column] + top_detections_df[column].astype(str).tolist() if not top_detections_df.empty else [column]
            top_det_sheet.set_column(col_idx, col_idx, min(max(len(v) for v in values) + 2, 32))

        detail_df.to_excel(writer, sheet_name="Detalle Tecnico", index=False)
        detail_sheet = writer.sheets["Detalle Tecnico"]
        detail_sheet.freeze_panes(1, 0)
        for col_idx, column in enumerate(detail_df.columns):
            detail_sheet.write(0, col_idx, column, header_fmt)
            values = [column] + detail_df[column].astype(str).tolist() if not detail_df.empty else [column]
            detail_sheet.set_column(col_idx, col_idx, min(max(len(v) for v in values) + 2, 34))
        detail_sheet.autofilter(0, 0, max(len(detail_df), 1), len(detail_df.columns) - 1)

        if not detail_df.empty:
            confidence_col = detail_df.columns.get_loc("Confianza")
            area_col = detail_df.columns.get_loc("Area (px)")
            row_count = len(detail_df)
            detail_sheet.conditional_format(
                1,
                confidence_col,
                row_count,
                confidence_col,
                {"type": "cell", "criteria": ">=", "value": 0.8, "format": workbook.add_format({"bg_color": "#DCFCE7"})},
            )
            detail_sheet.conditional_format(
                1,
                confidence_col,
                row_count,
                confidence_col,
                {
                    "type": "cell",
                    "criteria": "between",
                    "minimum": 0.6,
                    "maximum": 0.7999,
                    "format": workbook.add_format({"bg_color": "#DBEAFE"}),
                },
            )
            detail_sheet.conditional_format(
                1,
                confidence_col,
                row_count,
                confidence_col,
                {
                    "type": "cell",
                    "criteria": "between",
                    "minimum": 0.4,
                    "maximum": 0.5999,
                    "format": workbook.add_format({"bg_color": "#FEF3C7"}),
                },
            )
            detail_sheet.conditional_format(
                1,
                confidence_col,
                row_count,
                confidence_col,
                {"type": "cell", "criteria": "<", "value": 0.4, "format": workbook.add_format({"bg_color": "#FEE2E2"})},
            )
            detail_sheet.conditional_format(
                1,
                area_col,
                row_count,
                area_col,
                {"type": "data_bar", "bar_color": "#2563EB"},
            )

    return output.getvalue()


@router.get("/lab/images")
async def list_lab_images():
    return _list_image_files()


@router.get("/lab/annotated/{filename:path}")
async def get_lab_annotated_image(
    filename: str,
    request: Request,
    conf: float = Query(LAB_CONF, ge=0.0, le=1.0),
    min_area: int = Query(LAB_MIN_BBOX_AREA, ge=0),
):
    image_path = _resolve_image_path(filename)
    analysis = await _get_or_run_analysis(request, conf, min_area)
    image_analysis = next((img for img in analysis["images"] if img["filename"] == image_path.name), None)
    if image_analysis is None:
        raise HTTPException(status_code=404, detail="Analisis no disponible para la imagen")
    content = await asyncio.to_thread(_draw_detection_overlay, image_path, image_analysis)
    return Response(content=content, media_type="image/jpeg")


@router.get("/lab/images/{filename:path}")
async def get_lab_image(filename: str):
    image_path = _resolve_image_path(filename)
    return FileResponse(image_path)


@router.post("/lab/analyze-dataset")
async def analyze_lab_dataset(
    request: Request,
    conf: float = Query(LAB_CONF, ge=0.0, le=1.0),
    min_area: int = Query(LAB_MIN_BBOX_AREA, ge=0),
):
    return await _get_or_run_analysis(request, conf, min_area)


@router.get("/lab/report")
async def download_lab_report(
    request: Request,
    conf: float = Query(LAB_CONF, ge=0.0, le=1.0),
    min_area: int = Query(LAB_MIN_BBOX_AREA, ge=0),
):
    analysis = await _get_or_run_analysis(request, conf, min_area)
    report_bytes = await asyncio.to_thread(_build_report_bytes, analysis)
    filename = f"sipark_lab_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return StreamingResponse(
        BytesIO(report_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
