"""Pruebas del nucleo de deteccion.

Cubren la geometria y las reglas de fusion, que son deterministas y se pueden
comprobar con cajas sinteticas. No ejecutan YOLO: lo que se verifica aqui es la
logica que decide que deteccion sobrevive, no la calidad del detector.
"""

import unittest

import numpy as np

import detection as D


def det(x1, y1, x2, y2, score=0.9, truncated=False):
    return D.Detection((float(x1), float(y1), float(x2), float(y2)),
                       float(score), 3, "motorcycle", "tile", truncated)


class TestGeometria(unittest.TestCase):
    def test_interseccion_sobre_area(self):
        chica = (10, 10, 20, 20)
        grande = (0, 0, 100, 100)
        self.assertAlmostEqual(D._intersection_over_area(chica, grande), 1.0)
        self.assertAlmostEqual(D._intersection_over_area(grande, chica), 0.01)

    def test_iou_sin_solape(self):
        self.assertEqual(D._iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_iou_identicas(self):
        self.assertAlmostEqual(D._iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)


class TestSondeoDeEscena(unittest.TestCase):
    def test_sin_cajas_no_afirma_nada(self):
        scene = D.probe_scene(np.zeros((0, 4), np.float32), 1000)
        self.assertEqual(scene.viewpoint, "indeterminado")
        self.assertFalse(scene.reliable)

    def test_tamano_constante_es_vista_superior(self):
        # Motos del mismo tamano repartidas en altura: no hay perspectiva.
        boxes = np.array([[100, y, 160, y + 120] for y in range(50, 900, 40)],
                         dtype=np.float32)
        scene = D.probe_scene(boxes, 1000)
        self.assertTrue(scene.reliable)
        self.assertEqual(scene.viewpoint, "superior")
        self.assertLess(abs(scene.relative_slope), 0.55)

    def test_tamano_creciente_es_vista_oblicua(self):
        # El tamano crece con y: lo cercano se ve grande y esta abajo.
        boxes = []
        for y in range(50, 900, 40):
            h = 20 + y * 0.35
            boxes.append([100, y, 100 + h * 0.5, y + h])
        scene = D.probe_scene(np.array(boxes, np.float32), 1000)
        self.assertTrue(scene.reliable)
        self.assertEqual(scene.viewpoint, "oblicua")
        self.assertGreater(scene.relative_slope, 0.55)

    def test_motos_apinadas_no_permiten_medir(self):
        # Todas a la misma profundidad: la pendiente no es medible y no debe
        # darse por nula, porque eso equivaldria a afirmar vista superior.
        boxes = np.array([[x, 500, x + 60, 620] for x in range(50, 900, 70)],
                         dtype=np.float32)
        scene = D.probe_scene(boxes, 1000)
        self.assertEqual(scene.viewpoint, "indeterminado")
        self.assertFalse(scene.reliable)

    def test_alto_esperado_sigue_la_perspectiva(self):
        boxes = []
        for y in range(50, 900, 40):
            h = 20 + y * 0.35
            boxes.append([100, y, 100 + h * 0.5, y + h])
        scene = D.probe_scene(np.array(boxes, np.float32), 1000)
        cerca = scene.expected_height(850, 1000)
        lejos = scene.expected_height(100, 1000)
        self.assertGreater(cerca, lejos)


class TestPlanDeMosaicos(unittest.TestCase):
    def test_cubre_toda_la_imagen(self):
        scene = D.SceneModel(viewpoint="superior", median_side=60.0,
                             intercept=120.0, probe_count=20, reliable=True)
        tiles = D.plan_tiles(1600, 1200, scene, D.DetectParams())
        self.assertTrue(tiles)
        # Ningun mosaico se sale de la imagen.
        for x0, y0, x1, y1 in tiles:
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x1, 1600)
            self.assertLessEqual(y1, 1200)
        # Las esquinas quedan dentro de algun mosaico.
        for px, py in [(1, 1), (1599, 1), (1, 1199), (1599, 1199)]:
            self.assertTrue(any(x0 <= px <= x1 and y0 <= py <= y1
                                for x0, y0, x1, y1 in tiles))

    def test_respeta_el_presupuesto(self):
        scene = D.SceneModel(viewpoint="oblicua", median_side=12.0,
                             slope=0.02, intercept=8.0, probe_count=40,
                             reliable=True)
        params = D.DetectParams(max_tiles=20)
        tiles = D.plan_tiles(1600, 1200, scene, params)
        self.assertLessEqual(len(tiles), params.max_tiles)

    def test_objetos_pequenos_dan_mosaicos_mas_chicos(self):
        params = D.DetectParams()
        grande = D.SceneModel(median_side=300.0, intercept=600.0,
                              probe_count=20, reliable=True, viewpoint="superior")
        pequeno = D.SceneModel(median_side=40.0, intercept=80.0,
                               probe_count=20, reliable=True, viewpoint="superior")
        self.assertLess(len(D.plan_tiles(1600, 1200, grande, params)),
                        len(D.plan_tiles(1600, 1200, pequeno, params)))


class TestFusion(unittest.TestCase):
    def setUp(self):
        self.scene = D.SceneModel(reliable=False, median_side=120.0)
        self.params = D.DetectParams(truncated_conf=0.45, scale_tol=0.0)

    def test_conserva_dos_motos_vecinas(self):
        # Caso frecuente en filas apinadas: no deben fusionarse en una.
        dets = [det(100, 100, 160, 240), det(165, 100, 225, 240)]
        kept, _ = D.merge_detections(dets, self.scene, 1200, self.params)
        self.assertEqual(len(kept), 2)

    def test_descarta_la_parte_de_una_moto(self):
        entera = det(100, 100, 160, 240, score=0.9)
        rueda = det(105, 200, 155, 238, score=0.8)
        kept, dropped = D.merge_detections([entera, rueda], self.scene, 1200, self.params)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].box, entera.box)
        self.assertEqual(dropped["contenida"], 1)

    def test_descarta_la_caja_que_engloba_a_dos(self):
        # Reproduce el fallo observado: una caja de score bajo sobre dos motos
        # reales, que el NMS no quita porque solapa a medias con cada una.
        izq = det(398, 337, 449, 466, score=0.86)
        der = det(446, 335, 492, 466, score=0.82)
        envolvente = det(400, 335, 492, 467, score=0.25)
        kept, dropped = D.merge_detections([izq, der, envolvente], self.scene,
                                           1200, self.params)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped["envolvente"], 1)
        self.assertNotIn(envolvente.box, [k.box for k in kept])

    def test_la_caja_cortada_cede_ante_la_completa(self):
        completa = det(100, 100, 160, 240, score=0.7)
        cortada = det(100, 100, 160, 200, score=0.9, truncated=True)
        kept, _ = D.merge_detections([completa, cortada], self.scene, 1200, self.params)
        self.assertEqual(len(kept), 1)
        self.assertFalse(kept[0].truncated)

    def test_la_caja_cortada_sobrevive_si_nadie_la_cubre(self):
        cortada = det(100, 100, 160, 240, score=0.9, truncated=True)
        kept, _ = D.merge_detections([cortada], self.scene, 1200, self.params)
        self.assertEqual(len(kept), 1)

    def test_la_caja_cortada_debil_se_descarta(self):
        cortada = det(100, 100, 160, 240, score=0.30, truncated=True)
        kept, dropped = D.merge_detections([cortada], self.scene, 1200, self.params)
        self.assertEqual(kept, [])
        self.assertEqual(dropped["cortada_debil"], 1)

    def test_sin_geometria_medida_no_se_filtra_por_escala(self):
        # Una moto lejana es mucho mas chica que la mediana del primer plano:
        # filtrarla por tamano sin haber medido la perspectiva la eliminaria.
        params = D.DetectParams(scale_tol=2.0)
        lejana = det(100, 100, 130, 130, score=0.8)
        kept, dropped = D.merge_detections([lejana], self.scene, 1200, params)
        self.assertEqual(len(kept), 1)
        self.assertEqual(dropped["escala"], 0)

    def test_con_geometria_medida_si_se_filtra_por_escala(self):
        scene = D.SceneModel(reliable=True, median_side=120.0, slope=0.0,
                             intercept=120.0, viewpoint="superior")
        params = D.DetectParams(scale_tol=2.0)
        enorme = det(100, 100, 700, 700, score=0.8)
        kept, dropped = D.merge_detections([enorme], scene, 1200, params)
        self.assertEqual(kept, [])
        self.assertEqual(dropped["escala"], 1)

    def test_lista_vacia(self):
        kept, _ = D.merge_detections([], self.scene, 1200, self.params)
        self.assertEqual(kept, [])


class TestGeometriaReutilizada(unittest.TestCase):
    """La geometria guardada solo vale mientras la camara vea lo mismo.

    Arrastrar una medida vieja tras un cambio de encuadre degradaba el conteo
    durante muchos fotogramas, asi que se contrasta con la pasada completa.
    """

    def setUp(self):
        self.params = D.DetectParams()
        # Vista superior con motos de unos 120 px de alto.
        self.hint = D.SceneModel(viewpoint="superior", median_side=60.0,
                                 slope=0.0, intercept=120.0, reliable=True)

    def _boxes(self, alto):
        return np.array([[100, 400, 160, 400 + alto],
                         [200, 400, 260, 400 + alto],
                         [300, 400, 360, 400 + alto]], dtype=np.float32)

    def test_se_reutiliza_si_el_tamano_coincide(self):
        self.assertTrue(D._hint_matches(self.hint, self._boxes(120), 1000, self.params))

    def test_se_descarta_si_las_motos_se_ven_mucho_mas_grandes(self):
        self.assertFalse(D._hint_matches(self.hint, self._boxes(400), 1000, self.params))

    def test_se_descarta_si_se_ven_mucho_mas_pequenas(self):
        self.assertFalse(D._hint_matches(self.hint, self._boxes(30), 1000, self.params))

    def test_no_se_reutiliza_una_geometria_no_fiable(self):
        dudosa = D.SceneModel(viewpoint="indeterminado", median_side=60.0,
                              intercept=120.0, reliable=False)
        self.assertFalse(D._hint_matches(dudosa, self._boxes(120), 1000, self.params))

    def test_no_se_reutiliza_con_muy_pocas_cajas(self):
        pocas = np.array([[100, 400, 160, 520]], dtype=np.float32)
        self.assertFalse(D._hint_matches(self.hint, pocas, 1000, self.params))


class TestParametros(unittest.TestCase):
    def test_replace_no_muta_el_original(self):
        base = D.DetectParams()
        otro = base.replace(conf=0.5)
        self.assertEqual(base.conf, 0.25)
        self.assertEqual(otro.conf, 0.5)
        self.assertEqual(otro.class_names, base.class_names)


if __name__ == "__main__":
    unittest.main()
