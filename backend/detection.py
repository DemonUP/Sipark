"""Nucleo de deteccion adaptativo de Sipark.

Una sola implementacion, compartida por produccion y laboratorio, que estima la
geometria de la escena a partir de la propia imagen y elige la estrategia de
inferencia con esa medida, en lugar de asumir una camara aerea.

La estimacion mide el tamano de los objetos y como cambia con la altura de la
imagen. En una vista superior el tamano es casi constante; en una vista oblicua
decrece con la distancia. Esa pendiente clasifica el montaje y fija, por franja,
el tamano de mosaico con el que las motos ocupan una fraccion util del recorte.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence

import cv2
import numpy as np
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union


# ==========================
# PARAMETROS
# ==========================
@dataclass
class DetectParams:
    """Parametros de inferencia. Sin ajustes ocultos: el valor pedido es el usado."""

    # Umbrales del detector
    conf: float = 0.25
    iou: float = 0.60           # NMS interno de YOLO, por mosaico
    max_det: int = 3000

    # Sondeo de escena
    probe_imgsz: int = 960
    probe_conf: float = 0.12

    # Mosaicos
    tile_imgsz: int = 640
    target_obj_frac: float = 0.18   # lado del objeto / lado del mosaico
    min_tile: int = 320
    max_tile: int = 1600
    tile_overlap: float = 0.35
    bands: int = 3                  # franjas de profundidad
    scales: int = 2                 # niveles de mosaico, cada uno mas fino
    max_tiles: int = 60             # presupuesto de computo
    batch: int = 8

    # Fusion de resultados
    merge_iou: float = 0.55
    contain_thr: float = 0.85       # solape para considerar una caja incluida
    frag_area_ratio: float = 0.55   # ademas debe ser esta fraccion del area
    envelope_cover: float = 0.75    # caja cubierta por la union de otras
    edge_margin: int = 3            # px para considerar una caja cortada
    truncated_conf: float = 0.45     # una vista parcial exige mas confianza
    truncated_penalty: float = 0.35  # y cede la prioridad ante una completa
    hint_tol: float = 1.8            # desajuste que invalida una geometria dada
    scale_tol: float = 3.2          # razon admitida frente al tamano esperado

    # Clases
    class_names: tuple[str, ...] = ("motorcycle",)

    # Preproceso opcional (realce de contraste). Apagado por defecto.
    preprocess: bool = False

    def replace(self, **kw) -> "DetectParams":
        data = asdict(self)
        data.update(kw)
        data["class_names"] = tuple(data["class_names"])
        return DetectParams(**data)


@dataclass
class SceneModel:
    """Geometria estimada de la escena."""

    viewpoint: str = "indeterminado"   # superior | oblicua | indeterminado
    median_side: float = 0.0           # lado menor mediano, px
    slope: float = 0.0                 # d(alto) / d(y)
    intercept: float = 0.0
    relative_slope: float = 0.0        # pendiente normalizada, adimensional
    probe_count: int = 0
    reliable: bool = False

    def expected_height(self, cy: float, image_h: int) -> float:
        """Alto esperado de una moto cuyo centro esta en `cy`."""
        if not self.reliable:
            return max(self.median_side, 1.0)
        value = self.slope * float(cy) + self.intercept
        floor = max(4.0, 0.12 * self.median_side)
        return float(min(max(value, floor), float(image_h)))


@dataclass
class Detection:
    box: tuple[float, float, float, float]
    score: float
    cls: int
    cls_name: str
    source: str = "tile"    # tile | full
    truncated: bool = False  # la caja toca un borde interior del mosaico

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class DetectResult:
    detections: list[Detection] = field(default_factory=list)
    scene: SceneModel = field(default_factory=SceneModel)
    probe_scene: SceneModel | None = None
    scene_reused: bool = False
    tiles_used: int = 0
    candidates: int = 0
    truncated_candidates: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    inference_calls: int = 0


# ==========================
# GEOMETRIA
# ==========================
def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    bb = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes]
    idxs = cv2.dnn.NMSBoxes(
        bboxes=bb,
        scores=[float(s) for s in scores],
        score_threshold=0.0,
        nms_threshold=float(iou_thr),
        eta=1.0,
        top_k=0,
    )
    if idxs is None or len(idxs) == 0:
        return []
    return [int(i) for i in np.array(idxs).reshape(-1)]


def _intersection_over_area(a: Sequence[float], b: Sequence[float]) -> float:
    """Fraccion del area de `a` cubierta por `b`."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-6)
    return float(iw * ih / area_a)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return float(inter / max(area_a + area_b - inter, 1e-6))


# ==========================
# PREPROCESO
# ==========================
def enhance(img_bgr: np.ndarray) -> np.ndarray:
    """Realce de contraste local. Util con poca luz; puede anadir ruido de dia."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ==========================
# INFERENCIA
# ==========================
def _valid_class_ids(model, names: Iterable[str]) -> list[int]:
    wanted = {n.lower() for n in names}
    return sorted(k for k, v in model.names.items() if str(v).lower() in wanted)


def _predict(model, crops: list[np.ndarray], params: DetectParams,
             imgsz: int, conf: float, class_ids: list[int]):
    """Ejecuta el detector sobre una lista de recortes y devuelve cajas y scores."""
    if not crops:
        return []
    results = model.predict(
        crops,
        conf=float(conf),
        iou=float(params.iou),
        imgsz=int(imgsz),
        classes=class_ids or None,
        max_det=int(params.max_det),
        verbose=False,
        augment=False,
    )
    out = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            out.append((np.zeros((0, 4), np.float32),
                        np.zeros((0,), np.float32),
                        np.zeros((0,), np.int32)))
            continue
        out.append((
            r.boxes.xyxy.cpu().numpy().astype(np.float32),
            r.boxes.conf.cpu().numpy().astype(np.float32),
            r.boxes.cls.cpu().numpy().astype(np.int32),
        ))
    return out


# ==========================
# SONDEO DE ESCENA
# ==========================
def probe_scene(boxes: np.ndarray, image_h: int) -> SceneModel:
    """Estima tamano tipico y variacion con la profundidad a partir de un sondeo.

    Ajusta una recta sobre medianas por franja, no sobre cajas individuales, para
    que unas pocas deteccciones erroneas no dominen la pendiente.
    """
    scene = SceneModel(probe_count=int(len(boxes)))
    if len(boxes) == 0:
        return scene

    heights = boxes[:, 3] - boxes[:, 1]
    widths = boxes[:, 2] - boxes[:, 0]
    sides = np.minimum(widths, heights)
    scene.median_side = float(np.median(sides))

    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    median_h = float(np.median(heights))

    # Las franjas cubren el rango donde hay motos, no todo el alto de la imagen:
    # el cielo y el suelo vacio no aportan medidas y, contados como franjas sin
    # datos, impedirian medir la pendiente de una escena perfectamente oblicua.
    y_lo, y_hi = float(cy.min()), float(cy.max())
    y_range = y_hi - y_lo
    nbins = 5
    edges = np.linspace(y_lo, y_hi + 1e-6, nbins + 1)
    xs, ys, ws = [], [], []
    for i in range(nbins):
        sel = (cy >= edges[i]) & (cy < edges[i + 1])
        count = int(np.count_nonzero(sel))
        if count >= 2:
            xs.append(float((edges[i] + edges[i + 1]) / 2.0))
            ys.append(float(np.median(heights[sel])))
            ws.append(float(count))

    # El ajuste solo vale si las motos abarcan profundidad suficiente. Apinadas
    # en una linea la pendiente no es medible, y darla por nula equivaldria a
    # afirmar que la vista es superior.
    span = (max(xs) - min(xs)) if len(xs) >= 2 else 0.0
    if (len(xs) >= 3 and len(boxes) >= 8 and median_h > 1.0
            and span >= 0.12 * image_h and span >= 1.5 * median_h):
        slope, intercept = np.polyfit(np.array(xs), np.array(ys), 1, w=np.sqrt(ws))
        scene.slope = float(slope)
        scene.intercept = float(intercept)
        # Normaliza con el tramo realmente observado, que es donde el ajuste
        # tiene validez, y no con el alto completo de la imagen.
        scene.relative_slope = float(slope * max(y_range, 1.0) / median_h)
        scene.reliable = True
        # Una vista superior mantiene el tamano; una oblicua lo reduce con la
        # distancia. El signo depende de donde quede el fondo en la imagen.
        scene.viewpoint = "superior" if abs(scene.relative_slope) < 0.55 else "oblicua"
    else:
        # Geometria no medible: se conserva el tamano tipico como referencia,
        # pero sin declarar montaje ni habilitar el filtro de escala.
        scene.slope = 0.0
        scene.intercept = median_h
        scene.relative_slope = 0.0
        scene.reliable = False
        scene.viewpoint = "indeterminado"

    return scene


# ==========================
# PLAN DE MOSAICOS
# ==========================
def plan_tiles(image_w: int, image_h: int, scene: SceneModel,
               params: DetectParams) -> list[tuple[int, int, int, int]]:
    """Elige mosaicos por franja segun el tamano esperado de los objetos.

    Donde las motos se ven pequenas usa recortes pequenos, que el detector recibe
    ampliados; donde se ven grandes usa recortes amplios y evita gasto inutil.
    """
    if scene.probe_count == 0:
        # Sin referencia: cuadricula uniforme prudente.
        side = int(min(max(min(image_w, image_h) // 2, params.min_tile), params.max_tile))
        return _grid(image_w, image_h, side, 0, image_h, params.tile_overlap)

    # Con la geometria medida basta seguir la perspectiva por franjas. Cuando no
    # se pudo medir, el tamano tipico del sondeo esta sesgado al primer plano, y
    # una sola escala dejaria fuera las motos lejanas: se recorre varias.
    nbands = max(1, int(params.bands)) if scene.viewpoint == "oblicua" else 1
    nscales = 1 if scene.reliable else max(1, int(params.scales))
    band_edges = np.linspace(0.0, float(image_h), nbands + 1)

    def build(shrink: float) -> list[tuple[int, int, int, int]]:
        out: list[tuple[int, int, int, int]] = []
        for i in range(nbands):
            y_top, y_bottom = float(band_edges[i]), float(band_edges[i + 1])
            expected = scene.expected_height((y_top + y_bottom) / 2.0, image_h)
            for level in range(nscales):
                side = expected / max(params.target_obj_frac, 1e-3) / (2.0 ** level)
                side = min(max(side, params.min_tile), params.max_tile)
                # El factor del presupuesto se aplica despues del recorte: si se
                # aplicara antes, con objetos pequenos el minimo absorberia el
                # crecimiento y el limite de mosaicos nunca podria cumplirse.
                side = int(min(round(side * shrink), params.max_tile))
                out.extend(_grid(image_w, image_h, side, y_top, y_bottom,
                                 params.tile_overlap))
        return sorted(set(out))

    tiles = build(1.0)

    # Presupuesto de computo: si el plan excede el limite, agranda los mosaicos.
    guard = 0
    while len(tiles) > params.max_tiles and guard < 8:
        guard += 1
        rebuilt = build(1.35 ** guard)
        if len(rebuilt) >= len(tiles):
            break
        tiles = rebuilt

    return tiles


def _grid(image_w: int, image_h: int, side: int, y_top: float, y_bottom: float,
          overlap: float) -> list[tuple[int, int, int, int]]:
    """Cubre la banda [y_top, y_bottom) con recortes cuadrados de lado `side`."""
    side = int(min(side, max(image_w, image_h)))
    step = max(1, int(side * (1.0 - overlap)))
    tiles = []

    y_start = int(max(0, np.floor(y_top)))
    y_end = int(min(image_h, np.ceil(y_bottom)))
    if y_end <= y_start:
        return tiles

    ys = list(range(y_start, y_end, step)) or [y_start]
    xs = list(range(0, image_w, step)) or [0]

    for y0 in ys:
        y1 = min(image_h, y0 + side)
        y0a = max(0, y1 - side)
        if y0a >= y_end and y0 != y_start:
            continue
        for x0 in xs:
            x1 = min(image_w, x0 + side)
            x0a = max(0, x1 - side)
            tiles.append((x0a, y0a, x1, y1))
    return tiles


# ==========================
# FUSION
# ==========================
def _is_envelope(i: int, survivors: list[int], boxes: np.ndarray,
                  areas: np.ndarray, params: DetectParams) -> bool:
    """Indica si la caja `i` solo cubre objetos ya aceptados.

    Se exige que sea sensiblemente mayor que cada uno de los que solapa y que
    entre todos cubran casi toda su superficie. El area de la union se calcula
    de forma exacta: sumar los solapes por separado la sobreestimaria cuando los
    aceptados se solapan entre si, y descartaria motos reales.
    """
    overlapping = [j for j in survivors
                   if _intersection_over_area(boxes[j], boxes[i]) > 0.5
                   and areas[i] > 1.2 * areas[j]]
    if len(overlapping) < 2:
        return False

    target = shapely_box(*(float(v) for v in boxes[i]))
    if target.area <= 1e-6:
        return False
    union = unary_union([shapely_box(*(float(v) for v in boxes[j]))
                         for j in overlapping])
    return float(union.intersection(target).area / target.area) >= params.envelope_cover


def merge_detections(dets: list[Detection], scene: SceneModel, image_h: int,
                     params: DetectParams) -> tuple[list[Detection], dict[str, int]]:
    """Une resultados de todos los mosaicos y elimina duplicados y fragmentos."""
    dropped = {"nms": 0, "contenida": 0, "envolvente": 0, "escala": 0,
               "cortada_debil": 0}
    if not dets:
        return [], dropped

    # Una caja cortada aporta evidencia parcial del objeto, asi que se le pide
    # mas confianza que a una vista completa antes de entrar en la fusion.
    if params.truncated_conf > params.conf:
        kept_dets = [d for d in dets
                     if not d.truncated or d.score >= params.truncated_conf]
        dropped["cortada_debil"] = len(dets) - len(kept_dets)
        dets = kept_dets
        if not dets:
            return [], dropped

    boxes = np.array([d.box for d in dets], dtype=np.float32)
    scores = np.array([d.score for d in dets], dtype=np.float32)

    # El NMS resuelve los solapes por prioridad, y una caja cortada se penaliza:
    # es una vista parcial del objeto, asi que aunque el detector le asigne mas
    # confianza no debe desplazar a la caja que si lo abarca completo.
    priority = scores.copy()
    truncated_mask = np.array([d.truncated for d in dets], dtype=bool)
    priority[truncated_mask] *= (1.0 - params.truncated_penalty)

    keep = _nms(boxes, priority, params.merge_iou)
    dropped["nms"] = len(dets) - len(keep)
    if not keep:
        return [], dropped

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = sorted(keep, key=lambda i: float(priority[i]), reverse=True)

    # Se descarta la caja que es una parte de otra: muy incluida en ella y
    # bastante mas pequena. Exigir las dos condiciones evita borrar dos motos
    # vecinas que caen dentro de una caja grande, frecuente en filas apinadas.
    #
    # La caja cortada por el borde de su mosaico es una vista parcial por
    # definicion, asi que cede ante cualquier caja que la cubra, sin exigir
    # diferencia de tamano. Si ningun mosaico vio el objeto completo, sobrevive.
    survivors: list[int] = []
    for i in order:
        redundant = False
        for j in survivors:
            covered = _intersection_over_area(boxes[i], boxes[j])
            if covered < params.contain_thr:
                continue
            if areas[i] <= params.frag_area_ratio * areas[j]:
                redundant = True
                break
            if dets[i].truncated and not dets[j].truncated:
                redundant = True
                break
        if redundant:
            dropped["contenida"] += 1
            continue

        # Caso simetrico al fragmento: una caja mas grande que no aporta un
        # objeto nuevo, sino que engloba a varios ya aceptados. Sin esta regla
        # dos motos vecinas dejan una tercera deteccion sobre ambas, que el NMS
        # no quita porque solapa solo a medias con cada una.
        if _is_envelope(i, survivors, boxes, areas, params):
            dropped["envolvente"] += 1
            continue

        survivors.append(i)

    # Coherencia de escala frente al tamano esperado a esa profundidad. Solo se
    # aplica con la geometria medida; si no, el tamano de referencia vendria del
    # primer plano y castigaria justamente a las motos lejanas.
    final: list[int] = []
    for i in survivors:
        if not scene.reliable or params.scale_tol <= 1.0:
            final.append(i)
            continue
        x1, y1, x2, y2 = boxes[i]
        cy = float((y1 + y2) / 2.0)
        expected = scene.expected_height(cy, image_h)
        ratio = float(y2 - y1) / max(expected, 1e-6)
        if ratio > params.scale_tol or ratio < 1.0 / params.scale_tol:
            dropped["escala"] += 1
            continue
        final.append(i)

    final.sort(key=lambda i: float(scores[i]), reverse=True)
    return [dets[i] for i in final], dropped


# ==========================
# ENTRADA PRINCIPAL
# ==========================
def _hint_matches(hint: SceneModel, probe_boxes: np.ndarray, image_h: int,
                  params: DetectParams) -> bool:
    """Comprueba si una geometria ya medida sigue describiendo esta imagen.

    Reaprovechar la medida ahorra cuatro inferencias por fotograma, pero solo
    vale mientras la camara siga viendo lo mismo. Si se movio, cambio el
    encuadre o la imagen viene de otra fuente, arrastrar la medida vieja degrada
    el resultado durante muchos fotogramas. La pasada sobre la imagen completa
    se ejecuta siempre, asi que sirve de contraste sin costo adicional: se
    compara el tamano observado con el que la geometria predice a esa altura.
    """
    if not hint.reliable or len(probe_boxes) < 3:
        return False

    heights = probe_boxes[:, 3] - probe_boxes[:, 1]
    cy = (probe_boxes[:, 1] + probe_boxes[:, 3]) / 2.0
    observed = float(np.median(heights))
    expected = hint.expected_height(float(np.median(cy)), image_h)
    if observed <= 0.0 or expected <= 0.0:
        return False

    ratio = observed / expected
    return (1.0 / params.hint_tol) <= ratio <= params.hint_tol


def detect(model, img_bgr: np.ndarray, params: DetectParams | None = None,
           scene_hint: SceneModel | None = None) -> DetectResult:
    """Detecta motos adaptando la estrategia a la geometria medida de la imagen.

    `scene_hint` permite reaprovechar la geometria ya medida. En una camara fija
    el montaje no cambia entre fotogramas, asi que volver a sondear los
    cuadrantes en cada imagen solo gasta tiempo. La pasada sobre la imagen
    completa se mantiene siempre, porque ademas de medir aporta detecciones.
    """
    params = params or DetectParams()
    image = enhance(img_bgr) if params.preprocess else img_bgr
    H, W = image.shape[:2]
    class_ids = _valid_class_ids(model, params.class_names)
    id2name = {int(k): str(v) for k, v in model.names.items()}

    result = DetectResult()

    # 1) Sondeo. La pasada completa aporta las motos grandes del primer plano,
    #    pero por si sola tiende a no ver las lejanas, y con las medidas
    #    concentradas en una franja la perspectiva no se puede estimar. Cuatro
    #    cuadrantes anaden medidas a otras profundidades a un costo pequeno, y
    #    con ellas el plan de mosaicos se decide sobre una geometria medida en
    #    vez de recorrer varias escalas a ciegas.
    # Se infiere sola: mezclarla en un lote con recortes de otro tamano cambia
    # el letterbox y, con el, el resultado sobre la misma imagen.
    probe_boxes, probe_scores, probe_cls = _predict(
        model, [image], params, params.probe_imgsz, params.probe_conf, class_ids)[0]
    result.inference_calls += 1
    geometry_boxes = [probe_boxes]

    if scene_hint is not None and _hint_matches(scene_hint, probe_boxes, H, params):
        result.scene = scene_hint
        result.scene_reused = True
    else:
        quadrants = [
            (0, 0, (W + 1) // 2, (H + 1) // 2),
            (W // 2, 0, W, (H + 1) // 2),
            (0, H // 2, (W + 1) // 2, H),
            (W // 2, H // 2, W, H),
        ]
        quad_crops = [image[y0:y1, x0:x1] for x0, y0, x1, y1 in quadrants]
        quad_out = _predict(model, quad_crops, params, params.probe_imgsz,
                            params.probe_conf, class_ids)
        result.inference_calls += len(quad_crops)

        for (x0, y0, _, _), (qb, _, _) in zip(quadrants, quad_out):
            if len(qb):
                shifted = qb.copy()
                shifted[:, [0, 2]] += x0
                shifted[:, [1, 3]] += y0
                geometry_boxes.append(shifted)
        result.scene = probe_scene(np.vstack(geometry_boxes).astype(np.float32), H)

    probe_geometry = np.vstack(geometry_boxes).astype(np.float32)

    dets: list[Detection] = []
    for box, score, c in zip(probe_boxes, probe_scores, probe_cls):
        if float(score) < params.conf:
            continue
        dets.append(Detection(tuple(float(v) for v in box), float(score),
                              int(c), id2name.get(int(c), ""), "full"))

    # 2) Mosaicos elegidos con esa medida.
    tiles = plan_tiles(W, H, result.scene, params)
    result.tiles_used = len(tiles)

    for start in range(0, len(tiles), max(1, params.batch)):
        chunk = tiles[start:start + max(1, params.batch)]
        pairs = [(t, image[t[1]:t[3], t[0]:t[2]]) for t in chunk]
        pairs = [(t, c) for t, c in pairs if c.size > 0]
        if not pairs:
            continue
        batch = _predict(model, [c for _, c in pairs], params, params.tile_imgsz,
                         params.conf, class_ids)
        result.inference_calls += len(pairs)

        for ((x0, y0, x1, y1), _), (boxes, scores, clss) in zip(pairs, batch):
            for box, score, c in zip(boxes, scores, clss):
                bx1, by1, bx2, by2 = (float(v) for v in box)
                # Una caja pegada a un borde interior del mosaico esta cortada.
                # El borde que coincide con el de la imagen no corta nada.
                truncated = (
                    (bx1 <= params.edge_margin and x0 > 0)
                    or (by1 <= params.edge_margin and y0 > 0)
                    or (bx2 >= (x1 - x0) - params.edge_margin and x1 < W)
                    or (by2 >= (y1 - y0) - params.edge_margin and y1 < H)
                )
                dets.append(Detection(
                    (bx1 + x0, by1 + y0, bx2 + x0, by2 + y0),
                    float(score), int(c), id2name.get(int(c), ""), "tile",
                    truncated,
                ))

    result.candidates = len(dets)
    result.truncated_candidates = sum(1 for d in dets if d.truncated)

    # 3) Segunda estimacion de la geometria. Las cajas de los mosaicos cubren la
    #    profundidad mucho mejor que el sondeo, que solo alcanza el primer plano,
    #    asi que la pendiente medida aqui es la que vale para filtrar.
    if dets:
        refit_boxes = np.vstack([probe_geometry,
                                 np.array([d.box for d in dets], np.float32)])
        refit = probe_scene(refit_boxes, H)
        if not result.scene_reused and (refit.reliable or not result.scene.reliable):
            result.probe_scene = result.scene
            result.scene = refit

    # 4) Fusion: quita duplicados entre mosaicos, partes de moto y cortes.
    merged, dropped = merge_detections(dets, result.scene, H, params)
    dropped["cortadas_aceptadas"] = sum(1 for d in merged if d.truncated)
    result.dropped = dropped
    result.detections = sorted(merged, key=lambda d: d.score, reverse=True)
    return result
