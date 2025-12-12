import os, time, random, requests

API = "http://localhost:8000/api/ingest"
IMAGES_DIR = "./test_images"
INTERVAL_SEC = 5  # pruebas rápidas; cuando sea ESP32 real: 60

def main():
    imgs = [
        os.path.join(IMAGES_DIR, f)
        for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not imgs:
        print("No hay imágenes en", IMAGES_DIR)
        return

    while True:
        path = random.choice(imgs)
        with open(path, "rb") as f:
            files = {"file": (os.path.basename(path), f, "image/png")}
            r = requests.post(API, files=files)

        print("Sipark ingest:", os.path.basename(path), "->", r.status_code)
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
