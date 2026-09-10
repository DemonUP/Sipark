"""Servicio de monitoreo de Sipark.

La deteccion vive en `detection.py`, compartida con el laboratorio, y adapta su
estrategia a la geometria que mide en cada imagen. Aqui solo se resuelve la
asignacion de cada moto a una zona del parqueadero y la publicacion del estado.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

import detection
from debug_routes import router as debug_router
from lab_routes import router as lab_router

APP_NAME = "Sipark"
BASE_DIR = Path(__file__).resolve().parent


# ==========================
# CONFIG
# ==========================
MODEL_PATH = os.getenv("SIPARK_MODEL", "yolo11m.pt")

# Parametros de deteccion. El valor configurado es el que se usa: no hay
# minimos aplicados en silencio, para que un resultado sea reproducible.
PARAMS = detection.DetectParams(
    conf=float(os.getenv("SIPARK_CONF", "0.25")),
    truncated_conf=float(os.getenv("SIPARK_TRUNCATED_CONF", "0.45")),
    iou=float(os.getenv("SIPARK_IOU", "0.60")),
    probe_imgsz=int(os.getenv("SIPARK_PROBE_IMGSZ", "960")),
    tile_imgsz=int(os.getenv("SIPARK_TILE_IMGSZ", "640")),
    tile_overlap=float(os.getenv("SIPARK_OVERLAP", "0.35")),
    merge_iou=float(os.getenv("SIPARK_MERGE_IOU", "0.55")),
    max_tiles=int(os.getenv("SIPARK_MAX_TILES", "60")),
    preprocess=os.getenv("SIPARK_PREPROCESS", "0") == "1",
    class_names=("motorcycle", "bicycle") if os.getenv("SIPARK_INCLUDE_BICYCLE", "0") == "1"
    else ("motorcycle",),
)

# Fraccion minima de la caja dentro del poligono para asignarla a esa zona.
ASSIGN_MIN_RATIO = float(os.getenv("SIPARK_ASSIGN_RATIO", "0.08"))

# Cupos por zona. Sirve para informar saturacion, nunca para descartar motos:
# recortar el conteo al cupo ocultaria justamente la sobreocupacion y los
# errores de deteccion que hay que poder ver.
DEFAULT_ZONE_CAPACITY = int(os.getenv("SIPARK_ZONE_CAPACITY", "1"))

# Durante cuantos fotogramas se reaprovecha la geometria ya medida. En una
# camara fija el montaje no cambia, y reutilizarla ahorra cuatro inferencias por
# imagen: medido, 8,1 s la primera ingesta y 4,7 s las siguientes.
#
# Viene desactivado (0 = medir cada imagen) porque atar el plan de mosaicos a la
# geometria de un fotograma anterior cuesta detecciones cuando la escena no es
# exactamente la misma: en una prueba, 25 motos en vez de 28. Con una camara
# realmente fija ese costo desaparece y conviene activarlo para un feed en vivo.
SCENE_REFRESH_EVERY = int(os.getenv("SIPARK_SCENE_REFRESH", "0"))


# ==========================
# APP
# ==========================
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO(str(BASE_DIR / MODEL_PATH) if (BASE_DIR / MODEL_PATH).exists() else MODEL_PATH)


# ==========================
# ZONAS
# ==========================
def load_zones(path: Path | None = None):
    """Lee la definicion de zonas junto al modulo, no en el directorio actual."""
    cfg_path = path or (BASE_DIR / "zones.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    zones = [
        {
            "id": z["id"],
            "poly": Polygon(z["polygon"]),
            "capacity": int(z.get("capacity", DEFAULT_ZONE_CAPACITY)),
        }
        for z in cfg["zones"]
    ]
    view = str(cfg.get("view", "superior"))
    return int(cfg["image_width"]), int(cfg["image_height"]), zones, view


BASE_W, BASE_H, ZONES, ZONES_VIEW = load_zones()


def scale_polygon(poly: Polygon, w: int, h: int) -> Polygon:
    fx, fy = w / float(app.state.base_w), h / float(app.state.base_h)
    return Polygon([(x * fx, y * fy) for x, y in poly.exterior.coords])


def box_poly(x1: float, y1: float, x2: float, y2: float) -> Polygon:
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def assign_zone(box, zones_scaled) -> tuple[str | None, float]:
    """Asigna la caja a una zona: primero por centro, si no por mayor solape."""
    x1, y1, x2, y2 = box
    center = Point((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    for z in zones_scaled:
        if z["poly"].contains(center):
            return z["id"], 1.0

    bp = box_poly(x1, y1, x2, y2)
    area = bp.area if bp.area > 1e-6 else 1e-6
    best_id, best_ratio = None, 0.0
    for z in zones_scaled:
        if not z["poly"].intersects(bp):
            continue
        ratio = float(z["poly"].intersection(bp).area / area)
        if ratio > best_ratio:
            best_id, best_ratio = z["id"], ratio

    if best_ratio < ASSIGN_MIN_RATIO:
        return None, best_ratio
    return best_id, best_ratio


# ==========================
# STATE
# ==========================
app.state.lock = asyncio.Lock()
app.state.infer_lock = asyncio.Lock()
app.state.last_payload = None
app.state.last_image_jpg = None
app.state.model = model
app.state.model_names = model.names
app.state.base_w = BASE_W
app.state.base_h = BASE_H
app.state.zones = ZONES
app.state.zones_view = ZONES_VIEW
app.state.params = PARAMS
app.state.scene_hint = None
app.state.scene_age = 0
app.state.scene_shape = None
app.state.scale_polygon = scale_polygon
app.state.box_poly = box_poly


async def run_detection(img_bgr: np.ndarray, params: detection.DetectParams | None = None):
    """Ejecuta la inferencia fuera del hilo del servidor.

    El modelo no admite llamadas concurrentes, asi que se serializan con un
    candado propio; el trabajo pesado va a un hilo aparte para que la API siga
    respondiendo mientras se procesa una imagen.
    """
    async with app.state.infer_lock:
        hint = app.state.scene_hint
        if SCENE_REFRESH_EVERY <= 0 or app.state.scene_age >= SCENE_REFRESH_EVERY:
            hint = None
        # Otro tamano de imagen es otra fuente: la geometria anterior no aplica.
        shape = img_bgr.shape[:2]
        if app.state.scene_shape != shape:
            hint = None
        app.state.scene_shape = shape

        result = await asyncio.to_thread(detection.detect, model, img_bgr,
                                        params or app.state.params, hint)

        if result.scene_reused:
            app.state.scene_age += 1
        elif result.scene.reliable:
            app.state.scene_hint = result.scene
            app.state.scene_age = 0
        else:
            # Sin una medida fiable no se guarda nada: arrastrar una geometria
            # dudosa afectaria a todos los fotogramas siguientes.
            app.state.scene_hint = None
            app.state.scene_age = 0

        return result


app.state.run_detection = run_detection


# ==========================
# ENDPOINTS
# ==========================
@app.get("/api/health")
async def health():
    from dataclasses import asdict

    return {
        "ok": True,
        "app": APP_NAME,
        "model": MODEL_PATH,
        "zones": len(ZONES),
        "zones_view": ZONES_VIEW,
        "params": asdict(app.state.params),
        "assign_min_ratio": ASSIGN_MIN_RATIO,
        "zone_capacity_default": DEFAULT_ZONE_CAPACITY,
    }


@app.get("/api/last")
async def last():
    async with app.state.lock:
        return {"ok": True, "data": app.state.last_payload}


@app.get("/api/last-image")
async def last_image():
    async with app.state.lock:
        if app.state.last_image_jpg is None:
            return Response(status_code=404)
        return Response(content=app.state.last_image_jpg, media_type="image/jpeg")


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...), conf: float | None = None):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "No pude leer la imagen"}

    h, w = img.shape[:2]
    params = app.state.params if conf is None else app.state.params.replace(conf=float(conf))

    started = time.perf_counter()
    result = await run_detection(img, params)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    zones_scaled = [
        {"id": z["id"], "poly": scale_polygon(z["poly"], w, h), "capacity": z["capacity"]}
        for z in app.state.zones
    ]

    detections = []
    per_zone = {z["id"]: 0 for z in zones_scaled}
    unassigned = 0

    for det in result.detections:
        zid, ratio = assign_zone(det.box, zones_scaled)
        if zid is None:
            unassigned += 1
        else:
            per_zone[zid] += 1
        x1, y1, x2, y2 = det.box
        detections.append({
            "box": [x1, y1, x2, y2],
            "center": list(det.center),
            "zone": zid,
            "cls": det.cls,
            "cls_name": det.cls_name,
            "score": det.score,
            "assign_ratio": ratio,
            "truncated": det.truncated,
        })

    capacity = {z["id"]: z["capacity"] for z in zones_scaled}
    total_spaces = len(zones_scaled)
    occupied_spaces = sum(1 for v in per_zone.values() if v > 0)
    over_capacity = sorted(zid for zid, n in per_zone.items() if n > capacity[zid])

    # Las zonas solo describen cupos si la camara entrega la vista para la que
    # se dibujaron. Un poligono trazado sobre una vista superior no delimita un
    # cupo en una vista oblicua, aunque las cajas caigan dentro por estar la
    # rejilla extendida sobre toda la imagen: por eso no basta con mirar cuantas
    # motos quedan fuera, hay que comparar la geometria medida con la declarada.
    scene = result.scene
    assigned = len(detections) - unassigned
    assigned_share = (assigned / len(detections)) if detections else 0.0
    notes: list[str] = []

    if scene.reliable and scene.viewpoint != app.state.zones_view:
        notes.append(
            f"zones.json se definio para una vista '{app.state.zones_view}' y la "
            f"camara entrega una vista '{scene.viewpoint}'. La ocupacion por zona "
            f"no es representativa; redefina las zonas para esta camara."
        )

    if len(detections) >= 5 and assigned_share < 0.5:
        notes.append(
            "La mayoria de las motos detectadas cae fuera de las zonas: revise si "
            "zones.json corresponde al encuadre de esta camara."
        )

    zones_apply = not notes

    payload = {
        "app": APP_NAME,
        "timestamp": int(time.time()),
        "image_size": {"w": int(w), "h": int(h)},
        "totals": {
            "motos_detected": len(detections),
            "spaces_total": total_spaces,
            "spaces_occupied": occupied_spaces,
            "spaces_free": total_spaces - occupied_spaces,
            "motos_outside_zones": unassigned,
            "spaces_over_capacity": len(over_capacity),
        },
        "per_zone": per_zone,
        "zone_capacity": capacity,
        "zones_over_capacity": over_capacity,
        "detections": detections,
        # Geometria que el sistema midio solo, sin configurarla a mano.
        "scene": {
            "viewpoint": scene.viewpoint,
            "relative_slope": round(scene.relative_slope, 4),
            "median_side_px": round(scene.median_side, 1),
            "reliable": scene.reliable,
            "reused": result.scene_reused,
            "tiles_used": result.tiles_used,
        },
        "diagnostics": {
            "candidates": result.candidates,
            "truncated_candidates": result.truncated_candidates,
            "dropped": result.dropped,
            "inference_calls": result.inference_calls,
            "elapsed_ms": elapsed_ms,
            "assigned_share": round(assigned_share, 4),
            "zones_apply": zones_apply,
            "notes": notes,
        },
        "params": {**{k: v for k, v in vars(params).items()}, "assign_min_ratio": ASSIGN_MIN_RATIO},
    }

    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    # Imagen y resultado se publican juntos: si se hiciera por separado, un
    # cliente podria leer una imagen que no corresponde al estado mostrado.
    async with app.state.lock:
        if ok:
            app.state.last_image_jpg = jpg.tobytes()
        app.state.last_payload = payload

    return {"ok": True, "data": payload}


app.include_router(debug_router, prefix="/api")
app.include_router(lab_router, prefix="/api")
