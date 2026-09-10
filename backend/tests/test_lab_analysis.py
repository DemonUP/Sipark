"""Pruebas del laboratorio de imagenes.

El laboratorio ya no infiere por su cuenta: llama al nucleo comun
`detection.detect`, el mismo del monitoreo. Ese nucleo recorre la imagen por
mosaicos, asi que un modelo falso que devuelva siempre las mismas cajas sin
mirar el recorte no representa nada: cada mosaico repetiria la escena completa.

El falso de aqui si mira el recorte. La escena se dibuja como rectangulos
oscuros sobre fondo gris y el falso los localiza por contornos, en coordenadas
del recorte, como haria un detector. Asi la prueba ejercita el sondeo de
escena, el plan de mosaicos y la fusion, no solo el formato del resultado.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from io import BytesIO
from zipfile import ZipFile

import cv2
import numpy as np
import lab_routes as lab

BACKGROUND = 120
MOTO_GRAY = 25
MOTO_THRESHOLD = 70


class Tensor:
    """Minimo imprescindible de la interfaz de tensores que usa el nucleo."""

    def __init__(self, data):
        self.data = data

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class Boxes:
    def __init__(self, boxes, score, class_id):
        array = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        self.xyxy = Tensor(array)
        self.conf = Tensor(np.full(len(array), float(score), dtype=np.float32))
        self.cls = Tensor(np.full(len(array), int(class_id), dtype=np.int32))
        self.count = len(array)

    def __len__(self):
        return self.count


class FakeModel:
    """Detector falso: encuentra los rectangulos dibujados en cada recorte."""

    names = {3: "motorcycle"}

    def __init__(self, score=0.9, class_id=3):
        self.score = float(score)
        self.class_id = int(class_id)
        self.calls = []

    @property
    def params(self):
        # Compatibilidad: la ultima llamada al detector.
        return self.calls[-1] if self.calls else None

    def predict(self, images, **params):
        self.calls.append(params)
        crops = images if isinstance(images, list) else [images]
        results = []
        for crop in crops:
            boxes = []
            # El falso respeta la confianza pedida, igual que el detector real.
            if self.score >= float(params.get("conf", 0.0)) and crop is not None and crop.size:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                mask = (gray < MOTO_THRESHOLD).astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w * h >= 25:
                        boxes.append([float(x), float(y), float(x + w), float(y + h)])
            results.append(SimpleNamespace(boxes=Boxes(boxes, self.score, self.class_id)))
        return results


def write_scene(path: Path, boxes, size=(1000, 1000)) -> None:
    height, width = size
    image = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (MOTO_GRAY,) * 3, -1)
    cv2.imwrite(str(path), image)


# Siete motos separadas, del mismo tamano y a la misma altura: no hay
# profundidad medible, la geometria queda indeterminada.
FLAT_SCENE = [(60 + i * 130, 400, 150 + i * 130, 500) for i in range(7)]

# Escena en perspectiva: las cajas se hacen mas pequenas hacia arriba.
def oblique_scene():
    boxes = []
    for row, (y, side) in enumerate([(760, 170), (560, 120), (390, 80), (250, 50), (150, 30)]):
        for col in range(4):
            x = 60 + col * (side + 140)
            boxes.append((x, y, x + side, y + side))
    return boxes


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.override = patch.object(lab, "TEST_IMAGES_DIR", self.directory)
        self.override.start()
        write_scene(self.directory / "frame.png", FLAT_SCENE)

    def tearDown(self):
        self.override.stop()
        self.temp.cleanup()

    def test_detections_and_report(self):
        result = lab._analyze_dataset_sync(FakeModel(), 0.25, 5000)
        image = result["images"][0]
        # Sin recorte a seis motos: se reportan las siete de la escena.
        self.assertEqual(result["total_detections"], len(FLAT_SCENE))
        self.assertEqual(image["detection_count"], len(FLAT_SCENE))
        self.assertEqual(image["lighting_class"], "Brillo medio")
        self.assertEqual(image["predominant_angle"], "Compacta")
        # Los candidatos vienen de todos los mosaicos, asi que una misma moto
        # aporta varios; nunca pueden ser menos que las aceptadas.
        self.assertGreaterEqual(result["audit"]["candidates"], result["total_detections"])
        with ZipFile(BytesIO(lab._build_report_bytes(result))) as report:
            workbook = report.read("xl/workbook.xml").decode()
            self.assertIn("Metodologia", workbook)
            self.assertIn("Auditoria Filtros", workbook)

    def test_uses_shared_detection_core(self):
        """El laboratorio no debe tener inferencia propia."""
        with patch.object(lab.detection, "detect", wraps=lab.detection.detect) as detect:
            lab._analyze_dataset_sync(FakeModel(), 0.25, 5000)
        self.assertEqual(detect.call_count, 1)
        params = detect.call_args[0][2]
        self.assertIsInstance(params, lab.detection.DetectParams)
        self.assertEqual(params.conf, 0.25)

    def test_geometry_is_reported_per_image(self):
        result = lab._analyze_dataset_sync(FakeModel(), 0.25, 5000)
        image = result["images"][0]
        self.assertIn(image["viewpoint"], {"superior", "oblicua", "indeterminado"})
        self.assertGreater(image["tiles_used"], 0)
        self.assertGreater(image["median_side_px"], 0.0)
        self.assertIn("relative_slope", image)
        self.assertEqual(
            result["audit"]["tiles_used"],
            sum(item["tiles_used"] for item in result["images"]),
        )
        self.assertEqual(
            result["kpis"]["viewpoint_distribution"][image["viewpoint"]], 1
        )

    def test_perspective_scene_is_measured_and_covered(self):
        write_scene(self.directory / "frame.png", oblique_scene())
        result = lab._analyze_dataset_sync(FakeModel(), 0.25, 5000)
        image = result["images"][0]
        self.assertTrue(image["scene_reliable"])
        self.assertEqual(image["viewpoint"], "oblicua")
        self.assertNotEqual(image["relative_slope"], 0.0)
        # Las motos lejanas son de 30 px de lado: el lado minimo de 40 px que
        # aplicaba el laboratorio antes las habria descartado todas.
        small = [det for det in image["detections"] if det["bbox_area_px"] < 40 * 40]
        self.assertTrue(small)

    def test_low_confidence_is_not_silently_clamped(self):
        write_scene(self.directory / "frame.png", FLAT_SCENE[:1])
        model = FakeModel(score=0.15)
        result = lab._analyze_dataset_sync(model, 0.1, 5000)
        confs = {call["conf"] for call in model.calls}
        # La confianza pedida llega al detector tal cual. El sondeo de escena
        # usa su propia confianza porque sirve para medir la escala, no para
        # aceptar cajas; ninguna otra llamada puede exigir mas de lo pedido.
        self.assertIn(0.1, confs)
        self.assertEqual(confs - {0.1}, {lab._lab_params(0.1).probe_conf})
        self.assertEqual(result["total_detections"], 1)
        self.assertEqual(result["images"][0]["detections"][0]["confidence"], 0.15)

    def test_min_area_no_longer_filters(self):
        """min_area sigue en la API por compatibilidad, pero no descarta nada."""
        baseline = lab._analyze_dataset_sync(FakeModel(), 0.25, 0)
        huge = lab._analyze_dataset_sync(FakeModel(), 0.25, 20000)
        self.assertEqual(huge["total_detections"], baseline["total_detections"])
        self.assertEqual(huge["total_detections"], len(FLAT_SCENE))
        self.assertNotIn("min_area", huge["audit"]["rejected_by_reason"])
        self.assertFalse(huge["params"]["min_area_aplica"])
        self.assertEqual(huge["params"]["min_area"], 20000)

    def test_rejections_and_unreadable_images_are_accounted_for(self):
        (self.directory / "broken.jpg").write_bytes(b"not an image")
        result = lab._analyze_dataset_sync(FakeModel(), 0.25, 5000)
        self.assertEqual(result["audit"]["skipped_images"], ["broken.jpg"])
        self.assertEqual(result["audit"]["files_found"], 2)
        self.assertEqual(result["total_images"], 1)
        # Los descartes ahora los reporta la fusion del nucleo, y ya no
        # aparecen los filtros geometricos que aplicaba el laboratorio.
        reasons = result["audit"]["rejected_by_reason"]
        self.assertTrue(reasons)
        self.assertFalse(
            {"min_side", "min_area", "max_area", "aspect_ratio", "confidence"} & set(reasons)
        )
        # Las cortadas aceptadas no son un descarte y se reportan aparte.
        self.assertNotIn("cortadas_aceptadas", reasons)
        self.assertIn("accepted_truncated", result["audit"])
        for reason, count in reasons.items():
            self.assertEqual(
                count,
                sum(item["rejected_by_reason"].get(reason, 0) for item in result["images"]),
            )

    def test_missing_motorcycle_class_fails_explicitly(self):
        model = FakeModel()
        model.names = {0: "person"}
        with self.assertRaises(lab.HTTPException):
            lab._analyze_dataset_sync(model, 0.25, 5000)

    def test_empty_dataset_report(self):
        with patch.object(lab, "_list_image_files", return_value=[]):
            result = lab._analyze_dataset_sync(FakeModel(), 0.25, 5000)
        self.assertEqual(result["total_images"], 0)
        self.assertGreater(len(lab._build_report_bytes(result)), 0)


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        lab._analysis_cache.update(key=None, value=None)
        lab._analysis_inflight.clear()
        lab._analysis_state_lock = asyncio.Lock()

    async def test_image_changes_invalidate_cache(self):
        signature = [(("a.png", 100, 1),)]
        with patch.object(lab, "_dataset_signature", side_effect=lambda: signature[0]), patch.object(
            lab, "_get_lab_model", return_value=FakeModel()
        ), patch.object(lab, "_analyze_dataset_sync", return_value={"ok": True}) as analyze:
            await lab._get_or_run_analysis(None, 0.25, 5000)
            await lab._get_or_run_analysis(None, 0.25, 5000)
            self.assertEqual(analyze.call_count, 1)
            signature[0] = (("a.png", 100, 2),)
            await lab._get_or_run_analysis(None, 0.25, 5000)
            self.assertEqual(analyze.call_count, 2)

    async def test_cancelled_caller_does_not_cancel_shared_job(self):
        import threading
        entered = threading.Event()
        release = threading.Event()

        def analyze(*args):
            entered.set()
            release.wait(5)
            return {"ok": True}

        with patch.object(lab, "_dataset_signature", return_value=()), patch.object(
            lab, "_get_lab_model", return_value=FakeModel()
        ), patch.object(lab, "_analyze_dataset_sync", side_effect=analyze) as run:
            first = asyncio.create_task(lab._get_or_run_analysis(None, 0.25, 5000))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                first.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await first
                second = asyncio.create_task(lab._get_or_run_analysis(None, 0.25, 5000))
                release.set()
                self.assertEqual(await second, {"ok": True})
                self.assertEqual(run.call_count, 1)
            finally:
                release.set()


if __name__ == "__main__":
    unittest.main()
