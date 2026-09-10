import json
import tempfile
import unittest
from pathlib import Path

import evaluation as E


def box(x1, y1, x2, y2):
    return [float(x1), float(y1), float(x2), float(y2)]


# Cajas sinteticas: no se carga YOLO en ninguna prueba. Lo que se comprueba es la
# vara de medir, y esa debe ser correcta con independencia del detector.
BOX_A = box(0, 0, 100, 100)
BOX_A_SHIFT = box(10, 10, 110, 110)      # IoU alta con BOX_A
BOX_B = box(500, 500, 600, 600)
BOX_FAR = box(900, 900, 1000, 1000)


def annotation_set(records, origen=E.ORIGIN_PSEUDO):
    return E.AnnotationSet(
        conjunto="prueba", origen=origen,
        imagenes=[E.ImageAnnotation(**r) for r in records],
    )


class MatchingTests(unittest.TestCase):
    def test_coincidencia_perfecta(self):
        match = E.match_boxes([BOX_A, BOX_B], [BOX_A, BOX_B])
        self.assertEqual((match.tp, match.fp, match.fn), (2, 0, 0))
        self.assertEqual([m[2] for m in match.matches], [1.0, 1.0])

    def test_orden_de_las_cajas_no_altera_el_emparejamiento(self):
        match = E.match_boxes([BOX_B, BOX_A], [BOX_A, BOX_B])
        self.assertEqual((match.tp, match.fp, match.fn), (2, 0, 0))
        self.assertEqual(sorted(match.matches), [(0, 1, 1.0), (1, 0, 1.0)])

    def test_falso_positivo(self):
        match = E.match_boxes([BOX_A, BOX_FAR], [BOX_A])
        self.assertEqual((match.tp, match.fp, match.fn), (1, 1, 0))
        self.assertEqual(match.unmatched_pred, [1])

    def test_falso_negativo(self):
        match = E.match_boxes([BOX_A], [BOX_A, BOX_B])
        self.assertEqual((match.tp, match.fp, match.fn), (1, 0, 1))
        self.assertEqual(match.unmatched_gt, [1])

    def test_duplicado_sobre_la_misma_moto_es_falso_positivo(self):
        match = E.match_boxes([BOX_A, BOX_A_SHIFT], [BOX_A])
        self.assertEqual((match.tp, match.fp, match.fn), (1, 1, 0))
        # Gana la caja con IoU mayor, no la primera de la lista.
        self.assertEqual(match.matches[0][0], 0)

    def test_dos_predicciones_para_dos_anotaciones_solapadas(self):
        match = E.match_boxes([BOX_A, BOX_A_SHIFT], [BOX_A_SHIFT, BOX_A])
        self.assertEqual((match.tp, match.fp, match.fn), (2, 0, 0))

    def test_cajas_vacias_en_ambos_lados(self):
        match = E.match_boxes([], [])
        self.assertEqual((match.tp, match.fp, match.fn), (0, 0, 0))

    def test_sin_predicciones_todo_es_falso_negativo(self):
        match = E.match_boxes([], [BOX_A, BOX_B])
        self.assertEqual((match.tp, match.fp, match.fn), (0, 0, 2))

    def test_sin_anotaciones_todo_es_falso_positivo(self):
        match = E.match_boxes([BOX_A, BOX_B], [])
        self.assertEqual((match.tp, match.fp, match.fn), (0, 2, 0))

    def test_umbral_de_iou_decide_la_coincidencia(self):
        # IoU de BOX_A con BOX_A_SHIFT es 90*90 / (2*10000 - 8100).
        valor = E.iou_xyxy(BOX_A, BOX_A_SHIFT)
        self.assertAlmostEqual(valor, 8100.0 / 11900.0, places=6)
        self.assertEqual(E.match_boxes([BOX_A_SHIFT], [BOX_A], 0.5).tp, 1)
        self.assertEqual(E.match_boxes([BOX_A_SHIFT], [BOX_A], 0.8).tp, 0)

    def test_iou_de_cajas_disjuntas_es_cero(self):
        self.assertEqual(E.iou_xyxy(BOX_A, BOX_FAR), 0.0)

    def test_iou_de_caja_degenerada_no_divide_por_cero(self):
        self.assertEqual(E.iou_xyxy(box(0, 0, 0, 0), box(0, 0, 0, 0)), 0.0)


class MetricsTests(unittest.TestCase):
    def test_metricas_por_imagen(self):
        r = E.ImageResult(archivo="a.png", tp=3, fp=1, fn=1, n_pred=4, n_gt=4)
        self.assertAlmostEqual(r.precision, 0.75)
        self.assertAlmostEqual(r.recall, 0.75)
        self.assertAlmostEqual(r.f1, 0.75)
        self.assertEqual(r.count_error, 0)

    def test_imagen_vacia_sin_predicciones_es_acierto(self):
        r = E.ImageResult(archivo="a.png", n_pred=0, n_gt=0)
        self.assertEqual((r.precision, r.recall, r.f1), (1.0, 1.0, 1.0))

    def test_no_detectar_nada_con_motos_anotadas_no_es_precision_alta(self):
        r = E.ImageResult(archivo="a.png", fn=5, n_pred=0, n_gt=5)
        self.assertEqual(r.precision, 0.0)
        self.assertEqual(r.recall, 0.0)
        self.assertEqual(r.count_error, -5)

    def test_agregado_mae_y_sesgo_de_conteo(self):
        resultados = [
            E.ImageResult(archivo="a.png", tp=2, fp=2, fn=0, n_pred=4, n_gt=2),
            E.ImageResult(archivo="b.png", tp=2, fp=0, fn=2, n_pred=2, n_gt=4),
            E.ImageResult(archivo="c.png", tp=1, fp=0, fn=0, n_pred=1, n_gt=1),
        ]
        m = E.aggregate_metrics(resultados)
        self.assertEqual((m.tp, m.fp, m.fn), (5, 2, 2))
        self.assertAlmostEqual(m.precision, 5 / 7)
        self.assertAlmostEqual(m.recall, 5 / 7)
        self.assertAlmostEqual(m.count_mae, 4 / 3)
        # Sobreconteo y subconteo se cancelan: el sesgo no sustituye al MAE.
        self.assertAlmostEqual(m.count_bias, 0.0)

    def test_agregado_vacio(self):
        m = E.aggregate_metrics([])
        self.assertEqual((m.imagenes, m.tp, m.f1), (0, 0, 0.0))

    def test_evaluate_predictions_empareja_por_archivo(self):
        ann = annotation_set([
            {"archivo": "a.png", "cajas": [BOX_A, BOX_B], "revisado": True},
            {"archivo": "b.png", "cajas": [], "revisado": True},
        ], origen=E.ORIGIN_HUMAN)
        report = E.evaluate_predictions(ann, {
            "a.png": [BOX_A, BOX_FAR],
            "b.png": [],
            "sobra.png": [BOX_A],
        })
        self.assertTrue(report.valid)
        self.assertEqual(report.sin_anotacion, ["sobra.png"])
        self.assertEqual(report.metrics.tp, 1)
        self.assertEqual(report.metrics.fp, 1)
        self.assertEqual(report.metrics.fn, 1)
        self.assertAlmostEqual(report.metrics.count_mae, 0.0)

    def test_imagen_anotada_sin_prediccion_no_cuenta_como_fallo(self):
        ann = annotation_set([
            {"archivo": "a.png", "cajas": [BOX_A], "revisado": True},
            {"archivo": "b.png", "cajas": [BOX_A], "revisado": True},
        ], origen=E.ORIGIN_HUMAN)
        report = E.evaluate_predictions(ann, {"a.png": [BOX_A]})
        self.assertEqual(report.sin_prediccion, ["b.png"])
        self.assertEqual(report.metrics.imagenes, 1)
        self.assertAlmostEqual(report.metrics.f1, 1.0)


class ValidityTests(unittest.TestCase):
    def _report(self, revisado):
        ann = annotation_set([
            {"archivo": "a.png", "cajas": [BOX_A], "revisado": revisado},
        ])
        return E.evaluate_predictions(ann, {"a.png": [BOX_A]})

    def test_anotaciones_sin_revisar_producen_informe_no_valido(self):
        report = self._report(False)
        self.assertFalse(report.valid)
        self.assertEqual(report.unreviewed, ["a.png"])
        with self.assertRaises(E.AnnotationsNotReviewed):
            E.assert_reportable(report)

    def test_anotaciones_revisadas_producen_informe_valido(self):
        report = self._report(True)
        self.assertTrue(report.valid)
        self.assertEqual(report.unreviewed, [])
        E.assert_reportable(report)

    def test_una_sola_imagen_sin_revisar_invalida_el_conjunto(self):
        ann = annotation_set([
            {"archivo": "a.png", "cajas": [BOX_A], "revisado": True},
            {"archivo": "b.png", "cajas": [BOX_A], "revisado": False},
        ])
        report = E.evaluate_predictions(ann, {"a.png": [BOX_A], "b.png": [BOX_A]})
        self.assertFalse(report.valid)
        self.assertEqual(report.unreviewed, ["b.png"])

    def test_informe_sin_revision_queda_marcado_en_el_texto(self):
        texto = E.format_report(self._report(False))
        self.assertIn(E.INVALID_BANNER, texto)
        self.assertIn("[NO VALIDA]", texto)
        self.assertNotIn(E.VALID_BANNER, texto)

    def test_informe_revisado_no_lleva_marca_de_invalidez(self):
        texto = E.format_report(self._report(True))
        self.assertIn(E.VALID_BANNER, texto)
        self.assertNotIn("[NO VALIDA]", texto)

    def test_informe_en_json_declara_la_validez(self):
        data = self._report(False).to_dict()
        self.assertFalse(data["validas"])
        self.assertEqual(data["aviso"], E.INVALID_BANNER)
        self.assertEqual(data["sin_revisar"], ["a.png"])

    def test_sin_imagenes_evaluadas_tampoco_se_reporta(self):
        report = E.evaluate_predictions(annotation_set([]), {})
        self.assertFalse(report.valid)
        with self.assertRaises(E.AnnotationsNotReviewed):
            E.assert_reportable(report)

    def test_barrido_hereda_la_marca_de_invalidez(self):
        rows = [E.SweepRow(label="base", report=self._report(False),
                           params=E.D.DetectParams())]
        texto = E.format_sweep(rows)
        self.assertIn(E.INVALID_BANNER, texto)
        self.assertIn("[NO VALIDA]", texto)


class SweepTests(unittest.TestCase):
    def test_modo_eje_mueve_un_parametro_a_la_vez(self):
        base = E.D.DetectParams()
        configs = E.build_sweep_configs(base, {"conf": (0.2, 0.25, 0.3)}, "eje")
        self.assertEqual([c[0] for c in configs], ["base", "conf=0.2", "conf=0.3"])
        self.assertEqual(configs[1][1].conf, 0.2)
        # La base no se muta al derivar configuraciones.
        self.assertEqual(base.conf, 0.25)

    def test_modo_rejilla_recorre_el_producto(self):
        configs = E.build_sweep_configs(
            E.D.DetectParams(), {"conf": (0.2, 0.3), "merge_iou": (0.5, 0.6)},
            "rejilla")
        self.assertEqual(len(configs), 5)
        self.assertEqual(configs[0][0], "base")

    def test_max_configs_limita_el_gasto(self):
        configs = E.build_sweep_configs(max_configs=3)
        self.assertEqual(len(configs), 3)

    def test_modo_desconocido_falla(self):
        with self.assertRaises(ValueError):
            E.build_sweep_configs(mode="otro")

    def test_run_sweep_ordena_por_f1_con_predicciones_inyectadas(self):
        ann = annotation_set([
            {"archivo": "a.png", "cajas": [BOX_A, BOX_B], "revisado": True},
        ], origen=E.ORIGIN_HUMAN)
        base = E.D.DetectParams()
        configs = [("mala", base.replace(conf=0.9)), ("buena", base.replace(conf=0.1))]

        def predict(params, files):
            preds = E.PredictionSet()
            cajas = [BOX_A, BOX_B] if params.conf < 0.5 else [BOX_A]
            preds.boxes["a.png"] = cajas
            preds.scores["a.png"] = [0.9] * len(cajas)
            preds.seconds["a.png"] = 0.5
            return preds

        rows = E.run_sweep(None, ann, configs=configs, files=["a.png"],
                           verbose=False, predict=predict)
        self.assertEqual([r.label for r in rows], ["buena", "mala"])
        self.assertAlmostEqual(rows[0].report.metrics.f1, 1.0)
        self.assertTrue(rows[0].valid)
        self.assertAlmostEqual(rows[0].report.metrics.seconds_per_image, 0.5)


class AnnotationIoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "conjunto.json"

    def tearDown(self):
        self.temp.cleanup()

    def test_ida_y_vuelta(self):
        data = annotation_set([
            {"archivo": "a.png", "ancho": 640, "alto": 480,
             "cajas": [BOX_A], "revisado": True, "nota": "ok"},
            {"archivo": "b.png", "ancho": 640, "alto": 480, "cajas": []},
        ])
        E.save_annotations(data, self.path)
        cargado = E.load_annotations(self.path)
        self.assertEqual([a.archivo for a in cargado.imagenes], ["a.png", "b.png"])
        self.assertEqual(cargado.imagenes[0].cajas, [BOX_A])
        self.assertTrue(cargado.imagenes[0].revisado)
        self.assertFalse(cargado.imagenes[1].revisado)
        self.assertEqual(cargado.imagenes[1].cajas, [])
        self.assertTrue(cargado.creado)

    def test_lista_suelta_de_registros(self):
        self.path.write_text(json.dumps([{"archivo": "a.png", "boxes": [BOX_A]}]))
        cargado = E.load_annotations(self.path)
        self.assertEqual(cargado.imagenes[0].cajas, [BOX_A])
        self.assertFalse(cargado.imagenes[0].revisado)

    def test_caja_invertida_se_ordena(self):
        self.path.write_text(json.dumps(
            {"imagenes": [{"archivo": "a.png", "cajas": [[100, 100, 0, 0]]}]}))
        self.assertEqual(E.load_annotations(self.path).imagenes[0].cajas, [BOX_A])

    def test_revisado_no_booleano_no_cuenta_como_revisado(self):
        self.path.write_text(json.dumps(
            {"imagenes": [{"archivo": "a.png", "cajas": [], "revisado": "false"}]}))
        self.assertFalse(E.load_annotations(self.path).imagenes[0].revisado)

    def test_json_invalido(self):
        self.path.write_text("{no es json")
        with self.assertRaises(E.AnnotationError):
            E.load_annotations(self.path)

    def test_caja_incompleta(self):
        self.path.write_text(json.dumps(
            {"imagenes": [{"archivo": "a.png", "cajas": [[1, 2, 3]]}]}))
        with self.assertRaises(E.AnnotationError):
            E.load_annotations(self.path)

    def test_caja_degenerada(self):
        self.path.write_text(json.dumps(
            {"imagenes": [{"archivo": "a.png", "cajas": [[5, 5, 5, 9]]}]}))
        with self.assertRaises(E.AnnotationError):
            E.load_annotations(self.path)

    def test_archivo_con_ruta_se_rechaza(self):
        self.path.write_text(json.dumps(
            {"imagenes": [{"archivo": "../secreto.png", "cajas": []}]}))
        with self.assertRaises(E.AnnotationError):
            E.load_annotations(self.path)

    def test_archivo_duplicado_se_rechaza(self):
        self.path.write_text(json.dumps({"imagenes": [
            {"archivo": "a.png", "cajas": []},
            {"archivo": "a.png", "cajas": []},
        ]}))
        with self.assertRaises(E.AnnotationError):
            E.load_annotations(self.path)

    def test_archivo_ausente(self):
        with self.assertRaises(FileNotFoundError):
            E.load_annotations(self.path)


class PseudoLabelTests(unittest.TestCase):
    def test_pseudo_etiquetas_nacen_sin_revisar_y_marcadas(self):
        preds = E.PredictionSet(boxes={"a.png": [BOX_A], "b.png": []})
        data = E.build_pseudo_annotations(preds, {"a.png": (640, 480)})
        self.assertEqual(data.origen, E.ORIGIN_PSEUDO)
        self.assertIn("NO son verdad de referencia", data.nota)
        self.assertTrue(all(not a.revisado for a in data.imagenes))
        self.assertFalse(data.fully_reviewed)
        self.assertEqual(data.imagenes[0].ancho, 640)
        self.assertEqual(data.imagenes[1].cajas, [])

    def test_un_conjunto_de_pseudo_etiquetas_no_sirve_para_medir(self):
        preds = E.PredictionSet(boxes={"a.png": [BOX_A]})
        data = E.build_pseudo_annotations(preds, {"a.png": (640, 480)})
        report = E.evaluate_predictions(data, preds.boxes)
        # Acierto perfecto por construccion: el detector se compara consigo mismo.
        self.assertAlmostEqual(report.metrics.f1, 1.0)
        self.assertFalse(report.valid)
        with self.assertRaises(E.AnnotationsNotReviewed):
            E.assert_reportable(report)


if __name__ == "__main__":
    unittest.main()
