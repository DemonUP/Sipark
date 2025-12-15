import numpy as np
import cv2

from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import Response, HTMLResponse

router = APIRouter(tags=["debug"])


def _pts_from_poly(poly):
    # OpenCV a veces requiere Nx1x2
    return np.array(list(poly.exterior.coords), dtype=np.int32).reshape((-1, 1, 2))


@router.get("/debug/last-overlay")
async def debug_last_overlay(
    request: Request,
    thickness: int = 3,
    min_ratio: float = 0.10,
    show_zone_ids: bool = True,
    show_counts: bool = True,
    use_payload: bool = True,
):
    # 1) traer última imagen + payload
    async with request.app.state.lock:
        jpg_bytes = request.app.state.last_image_jpg
        payload = request.app.state.last_payload

    if jpg_bytes is None:
        return Response(status_code=404)

    img_np = np.frombuffer(jpg_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if img is None:
        return Response(status_code=500)

    h, w = img.shape[:2]

    # 2) zonas escaladas
    scale_polygon = request.app.state.scale_polygon
    zones = request.app.state.zones
    zones_scaled = [{"id": z["id"], "poly": scale_polygon(z["poly"], w, h)} for z in zones]

    # dibujar zonas
    for z in zones_scaled:
        poly = z["poly"]
        pts = _pts_from_poly(poly)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=thickness)

        if show_zone_ids:
            minx, miny, maxx, maxy = poly.bounds
            tx, ty = int(minx) + 8, int(miny) + 26
            cv2.putText(img, z["id"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # 3) detecciones encima
    #    Si use_payload=True, usamos lo que ya calculó /api/ingest (más rápido)
    if use_payload and payload and isinstance(payload, dict) and "detections" in payload:
        per_zone = payload.get("per_zone", {}) if isinstance(payload.get("per_zone", {}), dict) else {}

        # (opcional) contador por zona dentro de cada slot
        if show_counts and per_zone:
            for z in zones_scaled:
                zid = z["id"]
                n = int(per_zone.get(zid, 0))
                if n <= 0:
                    continue
                poly = z["poly"]
                minx, miny, maxx, maxy = poly.bounds
                tx, ty = int(minx) + 8, int(maxy) - 10
                cv2.putText(img, f"{n}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        for d in payload["detections"]:
            x1, y1, x2, y2 = d["box"]
            zid = d.get("zone", None)
            sc = float(d.get("score", 0.0))

            label_zone = "OUT" if zid is None else str(zid)

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            cv2.circle(img, (int(cx), int(cy)), 3, (0, 255, 255), -1)

            label = f"{label_zone} {sc:.2f}"
            cv2.putText(img, label, (int(x1), max(20, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    else:
        # fallback: si no hay payload, recalcula detecciones (más lento)
        tile_predict = request.app.state.tile_predict
        box_poly = request.app.state.box_poly

        boxes, cls, scores = tile_predict(img, conf=0.03, imgsz=1280, tile=640, overlap=0.35, nms_iou=0.65)

        for (x1, y1, x2, y2), sc in zip(boxes, scores):
            bp = box_poly(float(x1), float(y1), float(x2), float(y2))
            zid = None
            best_ratio = 0.0

            for z in zones_scaled:
                poly = z["poly"]
                if not poly.intersects(bp):
                    continue
                inter_area = poly.intersection(bp).area
                box_area = bp.area if bp.area > 1e-6 else 1e-6
                ratio = float(inter_area / box_area)
                if ratio > best_ratio:
                    best_ratio = ratio
                    zid = z["id"]

            label_zone = "OUT" if (zid is None or best_ratio < min_ratio) else str(zid)

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            label = f"{label_zone} {float(sc):.2f} r{best_ratio:.2f}"
            cv2.putText(img, label, (int(x1), max(20, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    ok, out = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        return Response(status_code=500)
    return Response(content=out.tobytes(), media_type="image/jpeg")


# opcional: una “página” para monitoreo que auto refresca (solo abres 1 URL)
@router.get("/debug/monitor", response_class=HTMLResponse)
async def debug_monitor():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8" />
      <title>Sipark Debug Monitor</title>
      <style>
        body{margin:0;background:#0b1020;color:#eaf0ff;font-family:system-ui}
        header{padding:12px 16px;background:rgba(255,255,255,.06);position:sticky;top:0}
        .wrap{padding:12px}
        img{max-width:98vw;border-radius:12px;border:1px solid rgba(255,255,255,.12)}
      </style>
    </head>
    <body>
      <header>
        <b>Sipark Debug</b> — refresca solo
      </header>
      <div class="wrap">
        <img id="img" src="/api/debug/last-overlay" />
      </div>
      <script>
        setInterval(()=>{
          const i = document.getElementById('img');
          i.src = "/api/debug/last-overlay?t=" + Date.now();
        }, 800);
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
