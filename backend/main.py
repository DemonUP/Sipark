import json
import time
import asyncio
import os
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from ultralytics import YOLO
from shapely.geometry import Polygon, Point

from debug_routes import router as debug_router


APP_NAME = "Sipark"

# ==========================
# CONFIG (sin entrenar)
# ==========================
MODEL_PATH = os.getenv("SIPARK_MODEL", "yolo11m.pt")   # CPU: yolo11s suele ir más liviano
DEFAULT_CONF = float(os.getenv("SIPARK_CONF", "0.012"))
PRED_IMGSZ = int(os.getenv("SIPARK_IMGSZ", "1536"))
PRED_IOU = float(os.getenv("SIPARK_IOU", "0.78"))
MAX_DET = int(os.getenv("SIPARK_MAX_DET", "3000"))

# ✅ mínimo de score aceptado (lo que pediste)
MIN_SCORE = float(os.getenv("SIPARK_MIN_SCORE", "0.10"))

# Tiling (clave en top-down)
TILE_SIZE = int(os.getenv("SIPARK_TILE", "1024"))
TILE_OVERLAP = float(os.getenv("SIPARK_OVERLAP", "0.55"))
GLOBAL_NMS_IOU = float(os.getenv("SIPARK_GLOBAL_NMS", "0.70"))

# Asignación a zona
ASSIGN_MIN_RATIO = float(os.getenv("SIPARK_ASSIGN_RATIO", "0.08"))

# TTA (augment) - en CPU lo dejamos apagado por defecto
USE_AUGMENT = os.getenv("SIPARK_AUGMENT", "0") == "1"

# Detectar todo y filtrar por nombre (robusto)
DETECT_ALL_THEN_FILTER = os.getenv("SIPARK_DETECT_ALL", "1") == "1"

# ⚠️ Si incluyes bicycle puedes inflar conteos. Recomendado: 0 para conteo de motos.
INCLUDE_BICYCLE = os.getenv("SIPARK_INCLUDE_BICYCLE", "0") == "1"

# ✅ NMS por zona (quita duplicados dentro del mismo puesto)
PER_ZONE_NMS_IOU = float(os.getenv("SIPARK_ZONE_NMS", "0.55"))

# ✅ CAP por espacio: para que E9 no sea 3 nunca si tu puesto solo admite 2
MAX_PER_ZONE = int(os.getenv("SIPARK_MAX_PER_ZONE", "2"))  # 2 = lo que estás pidiendo


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

# ==========================
# MODEL
# ==========================
model = YOLO(MODEL_PATH)

# Mapear por nombre para evitar IDs errados
ID2NAME = {k: v for k, v in model.names.items()}
NAME2ID = {v: k for k, v in model.names.items()}

VALID_NAMES = {"motorcycle"}
if INCLUDE_BICYCLE:
    VALID_NAMES.add("bicycle")

VALID_CLASS_IDS = [NAME2ID[n] for n in VALID_NAMES if n in NAME2ID]


# ==========================
# STATE
# ==========================
app.state.lock = asyncio.Lock()
app.state.last_payload = None
app.state.last_image_jpg = None


# ==========================
# ZONES
# ==========================
def load_zones():
    with open("zones.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    base_w = cfg["image_width"]
    base_h = cfg["image_height"]
    zones = [{"id": z["id"], "poly": Polygon(z["polygon"])} for z in cfg["zones"]]
    return base_w, base_h, zones


BASE_W, BASE_H, ZONES = load_zones()
app.state.base_w = BASE_W
app.state.base_h = BASE_H
app.state.zones = ZONES
app.state.model_names = model.names


def scale_point(x, y, w, h):
    return (x * (w / app.state.base_w), y * (h / app.state.base_h))


def scale_polygon(poly: Polygon, w, h):
    coords = list(poly.exterior.coords)
    scaled = [scale_point(x, y, w, h) for x, y in coords]
    return Polygon(scaled)


# ==========================
# PREPROCESS (drone/top-down)
# ==========================
def preprocess_drone(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.bilateralFilter(img_bgr, 5, 25, 25)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    img = cv2.filter2D(img, -1, kernel)
    return img


# ==========================
# PREDICT
# ==========================
def predict_boxes(img_bgr, conf=DEFAULT_CONF, imgsz=PRED_IMGSZ):
    conf_use = max(float(conf), float(MIN_SCORE))  # ✅ clamp mínimo

    r = model.predict(
        img_bgr,
        conf=conf_use,
        iou=float(PRED_IOU),
        imgsz=int(imgsz),
        classes=None if DETECT_ALL_THEN_FILTER else (VALID_CLASS_IDS if len(VALID_CLASS_IDS) else None),
        max_det=int(MAX_DET),
        verbose=False,
        augment=bool(USE_AUGMENT),
        agnostic_nms=False,
    )[0]

    if r.boxes is None or len(r.boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    cls = r.boxes.cls.cpu().numpy().astype(np.int32)
    scores = r.boxes.conf.cpu().numpy().astype(np.float32)

    # Filtrar por nombre (si detectamos todo)
    if DETECT_ALL_THEN_FILTER:
        keep = []
        for i, c in enumerate(cls):
            name = ID2NAME.get(int(c), "")
            if name in VALID_NAMES:
                keep.append(i)
        if len(keep) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float32),
            )
        boxes = boxes[keep]
        cls = cls[keep]
        scores = scores[keep]

    # ✅ filtro final por score >= MIN_SCORE
    keep2 = scores >= float(MIN_SCORE)
    if not np.any(keep2):
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    return boxes[keep2], cls[keep2], scores[keep2]


def nms_global(boxes_xyxy, scores, iou_thr=GLOBAL_NMS_IOU):
    if len(boxes_xyxy) == 0:
        return []

    boxes_xywh = []
    for x1, y1, x2, y2 in boxes_xyxy:
        boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

    idxs = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh,
        scores=scores.tolist(),
        score_threshold=0.0,
        nms_threshold=float(iou_thr),
        eta=1.0,
        top_k=0,
    )
    if idxs is None or len(idxs) == 0:
        return []
    return [int(i) for i in np.array(idxs).reshape(-1)]


def tile_predict(img_bgr, conf=DEFAULT_CONF, imgsz=PRED_IMGSZ,
                 tile=TILE_SIZE, overlap=TILE_OVERLAP, nms_iou=GLOBAL_NMS_IOU):
    H, W = img_bgr.shape[:2]

    # ✅ OPTIMIZACIÓN CPU: si cabe en un tile, 1 sola inferencia
    if H <= int(tile) and W <= int(tile):
        return predict_boxes(img_bgr, conf=conf, imgsz=imgsz)

    step = max(1, int(tile * (1 - overlap)))

    all_boxes, all_cls, all_scores = [], [], []

    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            x1 = min(W, x0 + tile)
            y1 = min(H, y0 + tile)
            x0a = max(0, x1 - tile)
            y0a = max(0, y1 - tile)

            crop = img_bgr[y0a:y1, x0a:x1]
            b, c, s = predict_boxes(crop, conf=conf, imgsz=imgsz)
            if len(b) == 0:
                continue

            b[:, [0, 2]] += x0a
            b[:, [1, 3]] += y0a

            all_boxes.append(b)
            all_cls.append(c)
            all_scores.append(s)

    if len(all_boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    boxes = np.vstack(all_boxes).astype(np.float32)
    cls = np.concatenate(all_cls).astype(np.int32)
    scores = np.concatenate(all_scores).astype(np.float32)

    keep = nms_global(boxes, scores, iou_thr=nms_iou)
    if len(keep) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )

    return boxes[keep], cls[keep], scores[keep]


def box_poly(x1, y1, x2, y2):
    return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


# Compartir con debug_routes
app.state.scale_polygon = scale_polygon
app.state.tile_predict = tile_predict
app.state.box_poly = box_poly


# ==========================
# ENDPOINTS
# ==========================
@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "model": MODEL_PATH,
        "valid_names": sorted(list(VALID_NAMES)),
        "valid_class_ids": VALID_CLASS_IDS,
        "params": {
            "conf": DEFAULT_CONF,
            "min_score": MIN_SCORE,
            "imgsz": PRED_IMGSZ,
            "pred_iou": PRED_IOU,
            "tile": TILE_SIZE,
            "overlap": TILE_OVERLAP,
            "global_nms_iou": GLOBAL_NMS_IOU,
            "zone_nms_iou": PER_ZONE_NMS_IOU,
            "max_per_zone": MAX_PER_ZONE,
            "augment": USE_AUGMENT,
            "detect_all_then_filter": DETECT_ALL_THEN_FILTER
        },
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
async def ingest(file: UploadFile = File(...), conf: float = DEFAULT_CONF):
    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "error": "No pude leer la imagen"}

    h, w = img.shape[:2]

    # guardar imagen original para debug
    ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if ok:
        async with app.state.lock:
            app.state.last_image_jpg = jpg.tobytes()

    # preproceso
    img_proc = preprocess_drone(img)

    # detección (tile_predict ya incluye "no tiling si cabe")
    boxes, cls, scores = tile_predict(
        img_proc,
        conf=float(conf),
        imgsz=int(PRED_IMGSZ),
        tile=int(TILE_SIZE),
        overlap=float(TILE_OVERLAP),
        nms_iou=float(GLOBAL_NMS_IOU),
    )

    # escalar zonas
    zones_scaled = [{"id": z["id"], "poly": scale_polygon(z["poly"], w, h)} for z in app.state.zones]

    # ==========================
    # 1) Asignación: guardamos detecciones (NO contamos aún)
    # ==========================
    detections = []

    for (x1, y1, x2, y2), c, sc in zip(boxes, cls, scores):
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)

        center_pt = Point(cx, cy)
        bp = box_poly(x1, y1, x2, y2)
        box_area = bp.area if bp.area > 1e-6 else 1e-6

        zid = None
        best_ratio = 0.0

        # 1) centro dentro de zona
        for z in zones_scaled:
            if z["poly"].contains(center_pt):
                zid = z["id"]
                best_ratio = 1.0
                break

        # 2) mejor intersección
        if zid is None:
            for z in zones_scaled:
                poly = z["poly"]
                if not poly.intersects(bp):
                    continue
                inter_area = poly.intersection(bp).area
                ratio = float(inter_area / box_area)
                if ratio > best_ratio:
                    best_ratio = ratio
                    zid = z["id"]

            if best_ratio < float(ASSIGN_MIN_RATIO):
                zid = None

        detections.append({
            "box": [x1, y1, x2, y2],
            "center": [cx, cy],
            "zone": zid,
            "cls": int(c),
            "cls_name": ID2NAME.get(int(c), ""),
            "score": float(sc),
            "assign_ratio": float(best_ratio),
        })

    # ==========================
    # 2) NMS POR ZONA (quita duplicados por tiling dentro del mismo puesto)
    # ==========================
    zone_to_idxs = {}
    for i, d in enumerate(detections):
        if d["zone"] is None:
            continue
        zone_to_idxs.setdefault(d["zone"], []).append(i)

    keep_det = set()
    for zid, idxs in zone_to_idxs.items():
        if len(idxs) == 1:
            keep_det.add(idxs[0])
            continue

        z_boxes = np.array([detections[i]["box"] for i in idxs], dtype=np.float32)
        z_scores = np.array([detections[i]["score"] for i in idxs], dtype=np.float32)

        kept_local = nms_global(z_boxes, z_scores, iou_thr=float(PER_ZONE_NMS_IOU))
        for k in kept_local:
            keep_det.add(idxs[k])

    detections = [
        d for i, d in enumerate(detections)
        if (d["zone"] is None) or (i in keep_det)
    ]

    # ==========================
    # 3) CAP POR ZONA: máximo N detecciones por puesto (por defecto 2)
    #    -> esto asegura que E9 no sea 3 si tu puesto solo admite 2
    # ==========================
    if MAX_PER_ZONE > 0:
        by_zone = {}
        unassigned_list = []
        for d in detections:
            if d["zone"] is None:
                unassigned_list.append(d)
            else:
                by_zone.setdefault(d["zone"], []).append(d)

        detections_limited = list(unassigned_list)

        for zid, ds in by_zone.items():
            if len(ds) <= MAX_PER_ZONE:
                detections_limited.extend(ds)
            else:
                # quedarse con las top por score
                ds_sorted = sorted(ds, key=lambda x: x["score"], reverse=True)
                detections_limited.extend(ds_sorted[:MAX_PER_ZONE])

        detections = detections_limited

    # ==========================
    # 4) Conteo final
    # ==========================
    per_zone = {z["id"]: 0 for z in zones_scaled}
    unassigned = 0

    for d in detections:
        if d["zone"] is None:
            unassigned += 1
        else:
            per_zone[d["zone"]] += 1

    total_spaces = len(zones_scaled)
    occupied_spaces = sum(1 for v in per_zone.values() if v > 0)
    free_spaces = int(total_spaces - occupied_spaces)

    payload = {
        "app": APP_NAME,
        "timestamp": int(time.time()),
        "image_size": {"w": int(w), "h": int(h)},
        "totals": {
            "motos_detected": int(len(detections)),  # ya post-proceso (dedupe + cap)
            "spaces_total": int(total_spaces),
            "spaces_occupied": int(occupied_spaces),
            "spaces_free": int(free_spaces),
            "motos_outside_zones": int(unassigned),
        },
        "per_zone": per_zone,
        "detections": detections,
        "params": {
            "conf": float(conf),
            "min_score": float(MIN_SCORE),
            "imgsz": int(PRED_IMGSZ),
            "tile": int(TILE_SIZE),
            "overlap": float(TILE_OVERLAP),
            "global_nms_iou": float(GLOBAL_NMS_IOU),
            "zone_nms_iou": float(PER_ZONE_NMS_IOU),
            "max_per_zone": int(MAX_PER_ZONE),
            "augment": bool(USE_AUGMENT),
            "detect_all_then_filter": bool(DETECT_ALL_THEN_FILTER),
            "valid_names": sorted(list(VALID_NAMES)),
        }
    }

    async with app.state.lock:
        app.state.last_payload = payload

    return {"ok": True, "data": payload}


app.include_router(debug_router, prefix="/api")
