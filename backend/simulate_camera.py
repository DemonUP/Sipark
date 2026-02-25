import os
import time
import cv2
import requests

API = "http://localhost:8000/api/ingest"
VIDEO_PATH = "../test_images/video.mp4"
PLAYBACK_SPEED = 1.0  # 1.0 = tiempo real, 2.0 = doble velocidad
JPEG_QUALITY = 85
REQUEST_TIMEOUT = 20


def resolve_video_path():
    base = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(base, VIDEO_PATH))
    return candidate


def frame_interval_for_real_time(source_fps: float) -> float:
    if source_fps <= 0:
        source_fps = 30.0
    return (1.0 / source_fps) / max(PLAYBACK_SPEED, 0.1)


def post_frame(frame, frame_index: int):
    ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        print(f"[WARN] No se pudo codificar frame {frame_index}")
        return

    files = {
        'file': (f'frame_{frame_index:06d}.jpg', jpg.tobytes(), 'image/jpeg')
    }

    try:
        r = requests.post(API, files=files, timeout=REQUEST_TIMEOUT)
        print(f"Sipark ingest: frame {frame_index:06d} -> {r.status_code}")
    except requests.RequestException as e:
        print(f"[ERROR] Error enviando frame {frame_index:06d}: {e}")


def main():
    video_path = resolve_video_path()
    if not os.path.exists(video_path):
        print(f"No existe el video: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = frame_interval_for_real_time(source_fps)

    print(f"Simulando cámara con video: {video_path}")
    print(f"FPS video: {source_fps:.2f} | Intervalo de envío: {interval:.3f}s")

    frame_index = 0
    while True:
        loop_start = time.time()
        ok, frame = cap.read()

        if not ok:
            print("Fin del video. Reiniciando desde el inicio...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        post_frame(frame, frame_index)
        frame_index += 1

        elapsed = time.time() - loop_start
        sleep_time = interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
