import json, time, asyncio
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from ultralytics import YOLO
from shapely.geometry import Polygon, Point


APP_NAME = "Sipark"

app = FastAPI(title=APP_NAME)

# CORS para que React consuma el API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción cambia por tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo liviano (nano)
model = YOLO("yolo11n.pt")  # ultralytics lo descargará si no está


def load_zones():
    with open("zones.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_w = cfg["image_width"]
    base_h = cfg["image_height"]
    zones = [{"id": z["id"], "poly": Polygon(z["polygon"])} for z in cfg["zones"]]
    return base_w, base_h, zones


BASE_W, BASE_H, ZONES = load_zones()


def scale_point(x, y, w, h):
    return (x * (w / BASE_W), y * (h / BASE_H))


def scale_polygon(poly: Polygon, w, h):
    coords = list(poly.exterior.coords)
    scaled = [scale_point(x, y, w, h) for x, y in coords]
    return Polygon(scaled)


# Última lectura guardada (para que el dashboard siempre lea lo más reciente)
_last = None
_last_image_jpg = None  # <-- guardamos la última imagen como JPG
_lock = asyncio.Lock()


@app.get("/api/health")
async def health():
    return {"ok": True, "app": APP_NAME}


@app.get("/api/last")
async def last():
    async with _lock:
        return {"ok": True, "data": _last}


@app.get("/api/last-image")
async def last_image():
    """
    Devuelve la última imagen recibida como JPEG.
    (Para que React la pueda mostrar en la "Vista de cámara")
    """
    async with _lock:
        if _last_image_jpg is None:
            return Response(status_code=404)
        return Response(content=_last_image_jpg, media_type="image/jpeg")


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...), conf: float = 0.25):
    global _last, _last_image_jpg

    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "No pude leer la imagen"}

    h, w = img.shape[:2]

    # Guardar JPG de la última imagen (calidad 85)
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok:
        _last_image_jpg = jpg.tobytes()

    # Detectar motos (COCO: motorcycle=3)
    res = model.predict(img, conf=conf, classes=[3], imgsz=640, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))

    # Escalar polígonos al tamaño real de imagen
    zones_scaled = [{"id": z["id"], "poly": scale_polygon(z["poly"], w, h)} for z in ZONES]

    per_zone = {z["id"]: 0 for z in zones_scaled}
    unassigned = 0
    detections = []

    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        p = Point(cx, cy)

        zid = None
        for z in zones_scaled:
            if z["poly"].contains(p):
                per_zone[z["id"]] += 1
                zid = z["id"]
                break

        if zid is None:
            unassigned += 1

        detections.append({
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "center": [float(cx), float(cy)],
            "zone": zid
        })

    total_spaces = len(zones_scaled)
    occupied_spaces = sum(1 for _, v in per_zone.items() if v > 0)
    free_spaces = total_spaces - occupied_spaces

    payload = {
        "app": APP_NAME,
        "timestamp": int(time.time()),
        "image_size": {"w": w, "h": h},
        "totals": {
            "motos_detected": int(len(boxes)),
            "spaces_total": total_spaces,
            "spaces_occupied": occupied_spaces,
            "spaces_free": free_spaces,
            "motos_outside_zones": int(unassigned)
        },
        "per_zone": per_zone,
        "detections": detections
    }

    async with _lock:
        _last = payload

    return {"ok": True, "data": payload}
