import asyncio
import logging
import os
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

logger = logging.getLogger("sipark.lab")

router = APIRouter(tags=["lab"])

LAB_CONF = 0.25
LAB_MIN_SCORE = 0.25
LAB_MIN_BBOX_AREA = 5000
LAB_IMGSZ = 640
LAB_NMS_IOU = 0.45
LAB_MIN_SIDE = 40
LAB_MIN_AREA_RATIO = 0.002
LAB_MAX_AREA_RATIO = 0.65
LAB_MIN_ASPECT_RATIO = 0.38
LAB_MAX_ASPECT_RATIO = 2.8
LAB_MAX_DETECTIONS_PER_IMAGE = 6

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
_analysis_inflight: dict[tuple[float, int], asyncio.Task] = {}


def _lighting_class(mean_brightness: float) -> str:
    if mean_brightness < 80:
        return "Noche"
    if mean_brightness <= 160:
        return "Tarde"
    return "Dia"


def _angle_class(aspect_ratio: float) -> str:
    if aspect_ratio > 1.3:
        return "Lateral"
    if aspect_ratio < 0.7:
        return "Trasera/Vertical"
    return "Frontal"


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


def _nms_boxes(boxes: list[list[float]], scores: list[float], iou_thr: float = LAB_NMS_IOU) -> list[int]:
    if not boxes:
        return []
    boxes_xywh = [[x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)] for x1, y1, x2, y2 in boxes]
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.0, nms_threshold=float(iou_thr))
    if idxs is None or len(idxs) == 0:
        return []
    return [int(i) for i in np.array(idxs).reshape(-1)]


def _contains_ratio(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    b_area = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter_area / b_area


def _should_keep_detection(
    score: float,
    width: float,
    height: float,
    bbox_area: float,
    image_area: float,
    aspect_ratio: float,
    min_area: int,
) -> bool:
    dynamic_min_area = max(float(min_area), image_area * LAB_MIN_AREA_RATIO)
    if score < LAB_MIN_SCORE:
        return False
    if width < LAB_MIN_SIDE or height < LAB_MIN_SIDE:
        return False
    if bbox_area < dynamic_min_area:
        return False
    if bbox_area > image_area * LAB_MAX_AREA_RATIO:
        return False
    if aspect_ratio < LAB_MIN_ASPECT_RATIO or aspect_ratio > LAB_MAX_ASPECT_RATIO:
        return False
    if score < 0.45 and bbox_area < dynamic_min_area * 1.6:
        return False
    return True


def _confidence_band(score: float) -> str:
    if score >= 0.8:
        return "Alta (>=0.80)"
    if score >= 0.6:
        return "Media (0.60-0.79)"
    return "Baja (<0.60)"


def _analyze_dataset_sync(model, conf: float, min_area: int) -> dict:
    image_files = _list_image_files()
    motorcycle_class_id = _find_motorcycle_class_id(model)

    images: list[dict] = []
    angle_distribution = {"Frontal": 0, "Lateral": 0, "Trasera/Vertical": 0}
    lighting_distribution = {"Dia": 0, "Tarde": 0, "Noche": 0}
    confidence_bands = {"Alta (>=0.80)": 0, "Media (0.60-0.79)": 0, "Baja (<0.60)": 0}
    total_detections = 0
    confidence_values: list[float] = []
    bbox_areas: list[float] = []
    images_with_detections = 0

    for filename in image_files:
        image_path = TEST_IMAGES_DIR / filename
        img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        image_h, image_w = img_bgr.shape[:2]
        image_area = float(image_w * image_h)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(gray.mean())
        lighting_class = _lighting_class(mean_brightness)
        lighting_distribution[lighting_class] += 1

        predict_kwargs = {
            "conf": max(float(conf), float(LAB_MIN_SCORE)),
            "imgsz": int(LAB_IMGSZ),
            "verbose": False,
            "iou": 0.50,
            "agnostic_nms": False,
            "classes": [motorcycle_class_id] if motorcycle_class_id is not None else None,
        }
        result = model.predict(img_bgr, **predict_kwargs)[0]

        raw_detections: list[dict] = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = result.boxes.conf.cpu().numpy().astype(np.float32)
            classes = result.boxes.cls.cpu().numpy().astype(np.int32)

            for bbox, score, class_id in zip(boxes, scores, classes):
                x1, y1, x2, y2 = [float(v) for v in bbox]
                width = max(0.0, x2 - x1)
                height = max(1.0, y2 - y1)
                bbox_area = float(width * height)
                aspect_ratio = float(width / height)
                class_name = str(model.names.get(int(class_id), class_id))

                if class_name.lower() != "motorcycle":
                    continue
                if not _should_keep_detection(
                    float(score), width, height, bbox_area, image_area, aspect_ratio, int(min_area)
                ):
                    continue

                raw_detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "bbox_area_px": bbox_area,
                        "confidence": float(score),
                        "class_name": class_name,
                        "aspect_ratio": aspect_ratio,
                        "angle_class": _angle_class(aspect_ratio),
                    }
                )

        kept_detections: list[dict] = []
        if raw_detections:
            keep_idxs = _nms_boxes([d["bbox"] for d in raw_detections], [d["confidence"] for d in raw_detections], LAB_NMS_IOU)
            nms_kept = [raw_detections[idx] for idx in keep_idxs]
            nms_kept.sort(key=lambda item: (item["confidence"], item["bbox_area_px"]), reverse=True)

            for det in nms_kept:
                if any(_contains_ratio(prev["bbox"], det["bbox"]) >= 0.82 for prev in kept_detections):
                    continue
                kept_detections.append(det)
                if len(kept_detections) >= LAB_MAX_DETECTIONS_PER_IMAGE:
                    break

        for idx, detection in enumerate(kept_detections):
            detection["detection_idx"] = idx
            detection["bbox"] = [round(v, 2) for v in detection["bbox"]]
            detection["bbox_area_px"] = round(detection["bbox_area_px"], 2)
            detection["confidence"] = round(detection["confidence"], 4)
            detection["aspect_ratio"] = round(detection["aspect_ratio"], 4)
            angle_distribution[detection["angle_class"]] += 1
            confidence_values.append(detection["confidence"])
            bbox_areas.append(detection["bbox_area_px"])
            confidence_bands[_confidence_band(detection["confidence"])] += 1

        detection_count = len(kept_detections)
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
            "min_area": int(min_area),
            "imgsz": LAB_IMGSZ,
            "min_score": LAB_MIN_SCORE,
            "nms_iou": LAB_NMS_IOU,
            "max_detections_per_image": LAB_MAX_DETECTIONS_PER_IMAGE,
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
        },
    }
    return analysis


async def _get_or_run_analysis(request: Request, conf: float, min_area: int) -> dict:
    cache_key = (round(float(conf), 4), int(min_area))

    async with _analysis_state_lock:
        if _analysis_cache["key"] == cache_key and _analysis_cache["value"] is not None:
            logger.info("Lab cache hit conf=%s min_area=%s", conf, min_area)
            return _analysis_cache["value"]

        existing_task = _analysis_inflight.get(cache_key)
        if existing_task is None:
            loop = asyncio.get_running_loop()
            logger.info("Lab analysis start conf=%s min_area=%s", conf, min_area)

            def _run_analysis_job() -> dict:
                model = _get_lab_model()
                return _analyze_dataset_sync(model, conf, min_area)

            existing_task = asyncio.ensure_future(
                loop.run_in_executor(_analysis_executor, _run_analysis_job)
            )
            _analysis_inflight[cache_key] = existing_task
        else:
            logger.info("Lab analysis join inflight conf=%s min_area=%s", conf, min_area)

    try:
        analysis = await existing_task
    finally:
        async with _analysis_state_lock:
            if _analysis_inflight.get(cache_key) is existing_task:
                _analysis_inflight.pop(cache_key, None)

    async with _analysis_state_lock:
        _analysis_cache["key"] = cache_key
        _analysis_cache["value"] = analysis

    logger.info(
        "Lab analysis done conf=%s min_area=%s detections=%s",
        conf,
        min_area,
        analysis.get("total_detections"),
    )
    return analysis


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
                "Angulo Predominante": image["predominant_angle"],
                "Resolucion": f"{image['image_w']} x {image['image_h']}",
                "Tiene Deteccion": "Si" if image["detection_count"] > 0 else "No",
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
                    "Angulo Estimado": detection["angle_class"],
                    "Brillo Medio": image["mean_brightness"],
                    "Iluminacion": image["lighting_class"],
                    "Cobertura Imagen (%)": image["bbox_coverage_pct"],
                }
            )

    image_df = pd.DataFrame(image_rows)
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
            "Angulo Estimado",
            "Brillo Medio",
            "Iluminacion",
            "Cobertura Imagen (%)",
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
            image_df.groupby("Angulo Predominante", dropna=False)
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
            columns=["Angulo Predominante", "Imagenes", "Detecciones", "ConfianzaMedia", "CoberturaMediaPct"]
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
                f"Parametros: conf={analysis['params']['conf']} | min_area={analysis['params']['min_area']} | "
                f"imgsz={analysis['params']['imgsz']} | nms_iou={analysis['params']['nms_iou']} | "
                f"max_det_img={analysis['params']['max_detections_per_image']}"
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
            f"validadas. {kpis['images_with_detections']} imagenes presentaron al menos una deteccion y "
            f"{kpis['images_without_detections']} no presentaron objetos validos. La confianza media fue "
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

        dashboard.write("A26", "Distribucion por Angulo", section_fmt)
        angle_labels = ["Frontal", "Lateral", "Trasera/Vertical"]
        for row_idx, label in enumerate(angle_labels, start=27):
            dashboard.write(f"A{row_idx}", label, label_fmt)
            dashboard.write(f"B{row_idx}", kpis["angle_distribution"][label], integer_fmt)

        dashboard.write("D26", "Distribucion por Iluminacion", section_fmt)
        lighting_labels = ["Dia", "Tarde", "Noche"]
        for row_idx, label in enumerate(lighting_labels, start=27):
            dashboard.write(f"D{row_idx}", label, label_fmt)
            dashboard.write(f"E{row_idx}", kpis["lighting_distribution"][label], integer_fmt)

        angle_chart = workbook.add_chart({"type": "column"})
        angle_chart.add_series(
            {
                "name": "Angulo",
                "categories": "=Dashboard!$A$27:$A$29",
                "values": "=Dashboard!$B$27:$B$29",
                "fill": {"color": "#2563EB"},
                "border": {"none": True},
            }
        )
        angle_chart.set_title({"name": "Distribucion por angulo"})
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

        angle_summary_df.to_excel(writer, sheet_name="Analisis Angulo", index=False)
        angle_sheet = writer.sheets["Analisis Angulo"]
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
