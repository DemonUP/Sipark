"""Arnes de evaluacion del detector de motos de Sipark.

Sin anotaciones de referencia validadas por una persona nadie puede afirmar que
un cambio mejora la exactitud: comparar el detector contra sus propias salidas es
circular y solo mide estabilidad. Este modulo separa las dos cosas de forma
explicita, y se niega a presentar como exactitud una medida tomada contra
pseudo-etiquetas.

FORMATO DE ANOTACION
--------------------
Un archivo JSON por conjunto, en `backend/annotations/` (por defecto
`test_images.json`). Estructura:

    {
      "version": 1,
      "conjunto": "test_images",
      "creado": "2026-01-01T00:00:00",
      "actualizado": "2026-01-01T00:00:00",
      "origen": "pseudo-etiquetas",       // o "revision-humana"
      "nota": "texto libre",
      "imagenes": [
        {
          "archivo": "img1.png",
          "ancho": 1920,
          "alto": 1080,
          "cajas": [[x1, y1, x2, y2], ...],   // motos, pixeles, esquinas
          "revisado": false,                  // true solo si una persona lo valido
          "nota": ""
        }
      ]
    }

Campos:
  - `archivo`: nombre del archivo dentro del directorio de imagenes. Sin rutas.
  - `ancho` / `alto`: tamano en pixeles de la imagen anotada. Sirve para detectar
    que la imagen cambio despues de anotarla.
  - `cajas`: una caja por moto visible, en coordenadas absolutas [x1, y1, x2, y2]
    con x1 < x2 y y1 < y2. Lista vacia significa "imagen sin motos", que es un
    dato valido y distinto de "imagen sin anotar".
  - `revisado`: bandera de validacion humana. Mientras sea `false` el registro es
    una pseudo-etiqueta y las metricas que se calculen contra el NO son medidas
    de exactitud.
  - `nota`: comentario opcional del anotador.

USO
---
    python -m evaluation pre-annotate      # pseudo-etiquetas para corregir a mano
    python -m evaluation evaluate          # metricas contra las anotaciones
    python -m evaluation sweep             # barrido de parametros ordenado por F1
    python -m evaluation status            # estado de revision del conjunto

Como modulo:

    import evaluation as E
    ann = E.load_annotations(E.DEFAULT_ANNOTATIONS)
    rep = E.evaluate_predictions(ann, {"img1.png": [[10, 10, 50, 50]]})
    E.assert_reportable(rep)               # lanza si falta revision humana
    print(E.format_report(rep))
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2

import detection as D


# ==========================
# RUTAS Y CONSTANTES
# ==========================
BASE_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = BASE_DIR / "test_images"
ANNOTATIONS_DIR = BASE_DIR / "annotations"
DEFAULT_ANNOTATIONS = ANNOTATIONS_DIR / "test_images.json"
DEFAULT_WEIGHTS = BASE_DIR / "yolo11m.pt"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
ANNOTATION_VERSION = 1

ORIGIN_PSEUDO = "pseudo-etiquetas"
ORIGIN_HUMAN = "revision-humana"

# Texto unico para marcar metricas sin respaldo humano. Se usa en todas las
# salidas para que no exista una forma de leer el informe sin ver el aviso.
INVALID_BANNER = (
    "METRICAS NO VALIDAS (PROVISIONALES): las anotaciones no estan revisadas por "
    "una persona. Son pseudo-etiquetas del propio detector, asi que estos numeros "
    "miden coincidencia consigo mismo, no exactitud."
)
VALID_BANNER = "Metricas validas: todas las imagenes usadas tienen revision humana."

PSEUDO_NOTE = (
    "Pseudo-etiquetas generadas por el detector. NO son verdad de referencia: "
    "corregir a mano y marcar revisado=true antes de medir exactitud."
)


class AnnotationsNotReviewed(RuntimeError):
    """Se pidio un informe de exactitud sobre anotaciones sin revision humana."""


class AnnotationError(ValueError):
    """El archivo de anotaciones no cumple el formato documentado."""


# ==========================
# MODELO DE DATOS
# ==========================
@dataclass
class ImageAnnotation:
    archivo: str
    ancho: int = 0
    alto: int = 0
    cajas: list[list[float]] = field(default_factory=list)
    revisado: bool = False
    nota: str = ""

    @property
    def conteo(self) -> int:
        return len(self.cajas)

    def to_dict(self) -> dict:
        return {
            "archivo": self.archivo,
            "ancho": int(self.ancho),
            "alto": int(self.alto),
            "cajas": [[round(float(v), 1) for v in box] for box in self.cajas],
            "revisado": bool(self.revisado),
            "nota": self.nota,
        }


@dataclass
class AnnotationSet:
    conjunto: str = "test_images"
    origen: str = ORIGIN_PSEUDO
    nota: str = ""
    creado: str = ""
    actualizado: str = ""
    version: int = ANNOTATION_VERSION
    imagenes: list[ImageAnnotation] = field(default_factory=list)

    def by_file(self) -> dict[str, ImageAnnotation]:
        return {a.archivo: a for a in self.imagenes}

    def get(self, archivo: str) -> ImageAnnotation | None:
        return self.by_file().get(archivo)

    @property
    def reviewed(self) -> list[ImageAnnotation]:
        return [a for a in self.imagenes if a.revisado]

    @property
    def unreviewed(self) -> list[ImageAnnotation]:
        return [a for a in self.imagenes if not a.revisado]

    @property
    def fully_reviewed(self) -> bool:
        return bool(self.imagenes) and not self.unreviewed

    def to_dict(self) -> dict:
        return {
            "version": int(self.version),
            "conjunto": self.conjunto,
            "creado": self.creado,
            "actualizado": self.actualizado,
            "origen": self.origen,
            "nota": self.nota,
            "imagenes": [a.to_dict() for a in self.imagenes],
        }


# ==========================
# CARGA Y GUARDADO
# ==========================
def _coerce_box(raw, contexto: str) -> list[float]:
    """Normaliza una caja a [x1, y1, x2, y2] ordenada."""
    if isinstance(raw, dict):
        try:
            raw = [raw["x1"], raw["y1"], raw["x2"], raw["y2"]]
        except KeyError as exc:
            raise AnnotationError(f"{contexto}: caja sin la clave {exc}") from exc
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise AnnotationError(f"{contexto}: la caja debe tener 4 numeros, llego {raw!r}")
    try:
        x1, y1, x2, y2 = (float(v) for v in raw)
    except (TypeError, ValueError) as exc:
        raise AnnotationError(f"{contexto}: caja con valores no numericos {raw!r}") from exc
    # Se acepta la caja invertida y se ordena: un anotador puede arrastrar el
    # raton de derecha a izquierda y eso no es un error del dato.
    box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    if box[2] - box[0] <= 0 or box[3] - box[1] <= 0:
        raise AnnotationError(f"{contexto}: caja degenerada {raw!r}")
    return box


def _coerce_record(raw, indice: int) -> ImageAnnotation:
    if not isinstance(raw, dict):
        raise AnnotationError(f"registro {indice}: se esperaba un objeto JSON")
    archivo = raw.get("archivo") or raw.get("file") or raw.get("nombre")
    if not archivo or not isinstance(archivo, str):
        raise AnnotationError(f"registro {indice}: falta 'archivo'")
    # Nunca se admite una ruta: el conjunto es plano y un ".." apuntaria fuera.
    if archivo != Path(archivo).name:
        raise AnnotationError(f"registro {indice}: 'archivo' no puede ser una ruta ({archivo})")
    contexto = f"{archivo}"
    cajas_raw = raw.get("cajas", raw.get("boxes", []))
    if cajas_raw is None:
        cajas_raw = []
    if not isinstance(cajas_raw, (list, tuple)):
        raise AnnotationError(f"{contexto}: 'cajas' debe ser una lista")
    cajas = [_coerce_box(b, contexto) for b in cajas_raw]
    try:
        ancho = int(raw.get("ancho", raw.get("width", 0)) or 0)
        alto = int(raw.get("alto", raw.get("height", 0)) or 0)
    except (TypeError, ValueError) as exc:
        raise AnnotationError(f"{contexto}: tamano no numerico") from exc
    revisado = raw.get("revisado", raw.get("reviewed", False))
    if not isinstance(revisado, bool):
        # Una cadena "false" o un 0 no pueden pasar por revision: ante la duda,
        # el registro cuenta como no revisado.
        revisado = str(revisado).strip().lower() in {"true", "1", "si", "yes"}
    return ImageAnnotation(
        archivo=archivo, ancho=ancho, alto=alto, cajas=cajas,
        revisado=bool(revisado), nota=str(raw.get("nota", "") or ""),
    )


def load_annotations(path: str | os.PathLike) -> AnnotationSet:
    """Lee un archivo de anotaciones y valida el formato documentado."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"no existe el archivo de anotaciones {path}. "
            "Genera pseudo-etiquetas con 'python -m evaluation pre-annotate'."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnnotationError(f"{path}: JSON invalido ({exc})") from exc

    # Se tolera una lista suelta de registros para no perder trabajo hecho a mano
    # con un formato mas simple.
    if isinstance(raw, list):
        raw = {"imagenes": raw}
    if not isinstance(raw, dict):
        raise AnnotationError(f"{path}: se esperaba un objeto o una lista")

    registros = raw.get("imagenes", raw.get("images", []))
    if not isinstance(registros, (list, tuple)):
        raise AnnotationError(f"{path}: 'imagenes' debe ser una lista")

    imagenes: list[ImageAnnotation] = []
    vistos: set[str] = set()
    for i, item in enumerate(registros):
        record = _coerce_record(item, i)
        if record.archivo in vistos:
            raise AnnotationError(f"{path}: archivo duplicado {record.archivo}")
        vistos.add(record.archivo)
        imagenes.append(record)

    return AnnotationSet(
        conjunto=str(raw.get("conjunto", path.stem)),
        origen=str(raw.get("origen", ORIGIN_PSEUDO)),
        nota=str(raw.get("nota", "") or ""),
        creado=str(raw.get("creado", "") or ""),
        actualizado=str(raw.get("actualizado", "") or ""),
        version=int(raw.get("version", ANNOTATION_VERSION) or ANNOTATION_VERSION),
        imagenes=imagenes,
    )


def _dump_json(payload: dict) -> str:
    """Serializa con cada caja en una sola linea.

    El archivo se corrige a mano: cuatro numeros repartidos en seis lineas hacen
    ilegible una imagen con veinte motos.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    box_pattern = re.compile(
        r"\[\s*\n\s*(-?[\d.]+),\s*\n\s*(-?[\d.]+),\s*\n\s*(-?[\d.]+),"
        r"\s*\n\s*(-?[\d.]+)\s*\n\s*\]"
    )
    return box_pattern.sub(r"[\1, \2, \3, \4]", text)


def save_annotations(data: AnnotationSet, path: str | os.PathLike) -> Path:
    """Guarda el conjunto. Escritura atomica: anotar a mano cuesta horas."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    if not data.creado:
        data.creado = now
    data.actualizado = now
    payload = _dump_json(data.to_dict())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)
    return path


# ==========================
# EMPAREJAMIENTO
# ==========================
def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """Interseccion sobre union de dos cajas [x1, y1, x2, y2].

    Se implementa aqui a proposito, en lugar de reutilizar el helper interno del
    detector: la vara de medir no debe depender del codigo que mide, o un error
    en esa funcion quedaria invisible en la evaluacion.
    """
    ax1, ay1, ax2, ay2 = (float(v) for v in a)
    bx1, by1, bx2, by2 = (float(v) for v in b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


@dataclass
class MatchResult:
    matches: list[tuple[int, int, float]] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)
    unmatched_gt: list[int] = field(default_factory=list)

    @property
    def tp(self) -> int:
        return len(self.matches)

    @property
    def fp(self) -> int:
        return len(self.unmatched_pred)

    @property
    def fn(self) -> int:
        return len(self.unmatched_gt)


def match_boxes(pred_boxes: Sequence[Sequence[float]],
                gt_boxes: Sequence[Sequence[float]],
                iou_thr: float = 0.5,
                pred_scores: Sequence[float] | None = None) -> MatchResult:
    """Empareja predicciones con anotaciones de forma codiciosa por IoU.

    Se recorren los pares en orden de IoU descendente y se fija cada
    emparejamiento uno a uno. Asi una segunda caja sobre la misma moto queda como
    falso positivo, que es lo que interesa medir: el conteo duplicado.
    """
    pairs: list[tuple[float, float, int, int]] = []
    for i, p in enumerate(pred_boxes):
        score = float(pred_scores[i]) if pred_scores is not None else 0.0
        for j, g in enumerate(gt_boxes):
            value = iou_xyxy(p, g)
            if value >= iou_thr and value > 0.0:
                pairs.append((value, score, i, j))

    # El desempate por score y luego por indice hace el resultado determinista,
    # condicion para comparar dos configuraciones sin ruido de ordenacion.
    pairs.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for value, _score, i, j in pairs:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
        matches.append((i, j, float(value)))

    matches.sort(key=lambda t: t[0])
    return MatchResult(
        matches=matches,
        unmatched_pred=[i for i in range(len(pred_boxes)) if i not in used_pred],
        unmatched_gt=[j for j in range(len(gt_boxes)) if j not in used_gt],
    )


# ==========================
# METRICAS
# ==========================
def _precision(tp: int, fp: int, n_gt: int) -> float:
    if tp + fp > 0:
        return tp / (tp + fp)
    # Sin predicciones no hay precision que medir. Vale 1.0 solo si tampoco habia
    # nada que detectar; con motos anotadas, no acertar nada no es precision alta.
    return 1.0 if n_gt == 0 else 0.0


def _recall(tp: int, fn: int) -> float:
    if tp + fn > 0:
        return tp / (tp + fn)
    return 1.0


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


@dataclass
class ImageResult:
    archivo: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_pred: int = 0
    n_gt: int = 0
    revisado: bool = False
    seconds: float = 0.0

    @property
    def precision(self) -> float:
        return _precision(self.tp, self.fp, self.n_gt)

    @property
    def recall(self) -> float:
        return _recall(self.tp, self.fn)

    @property
    def f1(self) -> float:
        return _f1(self.precision, self.recall)

    @property
    def count_error(self) -> int:
        """Diferencia de conteo con signo: positivo si el detector cuenta de mas."""
        return self.n_pred - self.n_gt

    def to_dict(self) -> dict:
        return {
            "archivo": self.archivo, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "n_pred": self.n_pred, "n_gt": self.n_gt, "revisado": self.revisado,
            "precision": round(self.precision, 4), "recall": round(self.recall, 4),
            "f1": round(self.f1, 4), "error_conteo": self.count_error,
            "segundos": round(self.seconds, 3),
        }


@dataclass
class Metrics:
    imagenes: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_pred: int = 0
    n_gt: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    count_mae: float = 0.0
    count_bias: float = 0.0
    seconds_per_image: float = 0.0

    def to_dict(self) -> dict:
        return {
            "imagenes": self.imagenes, "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "n_pred": self.n_pred, "n_gt": self.n_gt,
            "precision": round(self.precision, 4), "recall": round(self.recall, 4),
            "f1": round(self.f1, 4), "mae_conteo": round(self.count_mae, 4),
            "sesgo_conteo": round(self.count_bias, 4),
            "segundos_por_imagen": round(self.seconds_per_image, 3),
        }


def aggregate_metrics(results: Iterable[ImageResult]) -> Metrics:
    """Agrega por deteccion (micro) el acierto y por imagen el error de conteo."""
    results = list(results)
    m = Metrics(imagenes=len(results))
    if not results:
        return m
    m.tp = sum(r.tp for r in results)
    m.fp = sum(r.fp for r in results)
    m.fn = sum(r.fn for r in results)
    m.n_pred = sum(r.n_pred for r in results)
    m.n_gt = sum(r.n_gt for r in results)
    m.precision = _precision(m.tp, m.fp, m.n_gt)
    m.recall = _recall(m.tp, m.fn)
    m.f1 = _f1(m.precision, m.recall)
    m.count_mae = sum(abs(r.count_error) for r in results) / len(results)
    m.count_bias = sum(r.count_error for r in results) / len(results)
    m.seconds_per_image = sum(r.seconds for r in results) / len(results)
    return m


@dataclass
class EvaluationReport:
    metrics: Metrics = field(default_factory=Metrics)
    images: list[ImageResult] = field(default_factory=list)
    iou_thr: float = 0.5
    valid: bool = False
    unreviewed: list[str] = field(default_factory=list)
    sin_prediccion: list[str] = field(default_factory=list)
    sin_anotacion: list[str] = field(default_factory=list)
    origen: str = ORIGIN_PSEUDO

    @property
    def banner(self) -> str:
        return VALID_BANNER if self.valid else INVALID_BANNER

    def to_dict(self) -> dict:
        return {
            "validas": self.valid,
            "aviso": self.banner,
            "iou": self.iou_thr,
            "origen_anotaciones": self.origen,
            "sin_revisar": self.unreviewed,
            "sin_prediccion": self.sin_prediccion,
            "sin_anotacion": self.sin_anotacion,
            "agregado": self.metrics.to_dict(),
            "por_imagen": [r.to_dict() for r in self.images],
        }


def evaluate_predictions(annotations: AnnotationSet,
                         predictions: dict[str, Sequence[Sequence[float]]],
                         iou_thr: float = 0.5,
                         scores: dict[str, Sequence[float]] | None = None,
                         seconds: dict[str, float] | None = None) -> EvaluationReport:
    """Compara predicciones contra anotaciones y devuelve el informe.

    El informe se calcula siempre, pero nace marcado como no valido si alguna
    imagen usada no tiene revision humana. La decision de mostrarlo o negarse es
    de quien lo consume (`assert_reportable`), no de este calculo.
    """
    report = EvaluationReport(iou_thr=float(iou_thr), origen=annotations.origen)
    scores = scores or {}
    seconds = seconds or {}

    anotadas = annotations.by_file()
    report.sin_anotacion = sorted(set(predictions) - set(anotadas))

    for archivo, ann in sorted(anotadas.items()):
        if archivo not in predictions:
            report.sin_prediccion.append(archivo)
            continue
        pred = list(predictions[archivo])
        match = match_boxes(pred, ann.cajas, iou_thr, scores.get(archivo))
        report.images.append(ImageResult(
            archivo=archivo, tp=match.tp, fp=match.fp, fn=match.fn,
            n_pred=len(pred), n_gt=ann.conteo, revisado=ann.revisado,
            seconds=float(seconds.get(archivo, 0.0)),
        ))
        if not ann.revisado:
            report.unreviewed.append(archivo)

    report.metrics = aggregate_metrics(report.images)
    report.valid = bool(report.images) and not report.unreviewed
    return report


def assert_reportable(report: EvaluationReport) -> None:
    """Lanza si el informe no puede presentarse como medida de exactitud."""
    if report.valid:
        return
    if not report.images:
        raise AnnotationsNotReviewed(
            "no hay ninguna imagen con anotacion y prediccion: nada que medir."
        )
    raise AnnotationsNotReviewed(
        f"{len(report.unreviewed)} de {len(report.images)} imagenes evaluadas no "
        "tienen revision humana. " + INVALID_BANNER
    )


# ==========================
# INFORMES EN TEXTO
# ==========================
def _tag(report: EvaluationReport) -> str:
    return "" if report.valid else "[NO VALIDA] "


def format_report(report: EvaluationReport, per_image: bool = True) -> str:
    """Texto del informe. Si no hay revision humana, cada linea queda marcada."""
    m = report.metrics
    lines = ["=" * 78, report.banner, "=" * 78, ""]
    lines.append(f"IoU de emparejamiento: {report.iou_thr:.2f}")
    lines.append(f"Origen de las anotaciones: {report.origen}")
    lines.append(f"Imagenes evaluadas: {m.imagenes} "
                 f"(sin revisar: {len(report.unreviewed)})")
    if report.sin_prediccion:
        lines.append(f"Anotadas sin prediccion: {len(report.sin_prediccion)} "
                     f"-> {', '.join(report.sin_prediccion[:5])}")
    if report.sin_anotacion:
        lines.append(f"Predichas sin anotacion (ignoradas): {len(report.sin_anotacion)}")
    lines.append("")

    tag = _tag(report)
    lines.append(f"{tag}AGREGADO")
    lines.append(f"  {tag}VP={m.tp}  FP={m.fp}  FN={m.fn}  "
                 f"predichas={m.n_pred}  anotadas={m.n_gt}")
    lines.append(f"  {tag}precision={m.precision:.4f}  recall={m.recall:.4f}  "
                 f"F1={m.f1:.4f}")
    lines.append(f"  {tag}MAE de conteo={m.count_mae:.3f}  "
                 f"sesgo de conteo={m.count_bias:+.3f}")
    if m.seconds_per_image > 0:
        lines.append(f"  {tag}segundos por imagen={m.seconds_per_image:.2f}")

    if per_image and report.images:
        lines.append("")
        lines.append(f"{tag}POR IMAGEN")
        header = (f"  {'archivo':<44} {'rev':>3} {'VP':>4} {'FP':>4} {'FN':>4} "
                  f"{'pred':>5} {'anot':>5} {'F1':>7}")
        lines.append(header)
        for r in report.images:
            nombre = r.archivo if len(r.archivo) <= 44 else "..." + r.archivo[-41:]
            lines.append(f"  {nombre:<44} {'si' if r.revisado else 'NO':>3} "
                         f"{r.tp:>4} {r.fp:>4} {r.fn:>4} {r.n_pred:>5} "
                         f"{r.n_gt:>5} {r.f1:>7.4f}")

    lines.append("")
    lines.append(report.banner)
    return "\n".join(lines)


# ==========================
# DETECTOR
# ==========================
def list_images(images_dir: str | os.PathLike) -> list[str]:
    directory = Path(images_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"no existe el directorio de imagenes {directory}")
    return sorted(p.name for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def load_model(weights: str | os.PathLike = DEFAULT_WEIGHTS):
    """Carga YOLO. Import diferido: el arnes se importa tambien sin el modelo."""
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.is_file():
        raise FileNotFoundError(f"no existen los pesos {weights}")
    return YOLO(str(weights))


@dataclass
class PredictionSet:
    boxes: dict[str, list[list[float]]] = field(default_factory=dict)
    scores: dict[str, list[float]] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    scene: dict[str, str] = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)


def run_detector(model, files: Sequence[str],
                 images_dir: str | os.PathLike = TEST_IMAGES_DIR,
                 params: D.DetectParams | None = None,
                 verbose: bool = False) -> PredictionSet:
    """Ejecuta el detector sobre `files` y devuelve cajas, scores y tiempos."""
    params = params or D.DetectParams()
    directory = Path(images_dir)
    out = PredictionSet()
    for name in files:
        path = directory / name
        img = cv2.imread(str(path))
        if img is None:
            out.unreadable.append(name)
            continue
        started = time.perf_counter()
        res = D.detect(model, img, params)
        elapsed = time.perf_counter() - started
        out.boxes[name] = [[float(v) for v in d.box] for d in res.detections]
        out.scores[name] = [float(d.score) for d in res.detections]
        out.seconds[name] = elapsed
        out.scene[name] = res.scene.viewpoint
        if verbose:
            print(f"  {name}: {len(res.detections)} motos, "
                  f"vista={res.scene.viewpoint}, mosaicos={res.tiles_used}, "
                  f"{elapsed:.2f}s", flush=True)
    return out


# ==========================
# PRE-ANOTACION
# ==========================
def build_pseudo_annotations(predictions: PredictionSet,
                             sizes: dict[str, tuple[int, int]],
                             conjunto: str = "test_images") -> AnnotationSet:
    """Arma un conjunto de anotaciones con `revisado: false` en todas."""
    data = AnnotationSet(conjunto=conjunto, origen=ORIGIN_PSEUDO, nota=PSEUDO_NOTE)
    for name in sorted(predictions.boxes):
        ancho, alto = sizes.get(name, (0, 0))
        data.imagenes.append(ImageAnnotation(
            archivo=name, ancho=int(ancho), alto=int(alto),
            cajas=[list(b) for b in predictions.boxes[name]],
            revisado=False, nota=PSEUDO_NOTE,
        ))
    return data


def pre_annotate(model, images_dir: str | os.PathLike = TEST_IMAGES_DIR,
                 files: Sequence[str] | None = None,
                 params: D.DetectParams | None = None,
                 verbose: bool = True) -> AnnotationSet:
    directory = Path(images_dir)
    names = list(files) if files is not None else list_images(directory)
    sizes: dict[str, tuple[int, int]] = {}
    for name in names:
        img = cv2.imread(str(directory / name))
        if img is not None:
            sizes[name] = (int(img.shape[1]), int(img.shape[0]))
    predictions = run_detector(model, names, directory, params, verbose=verbose)
    return build_pseudo_annotations(predictions, sizes, conjunto=directory.name)


# ==========================
# BARRIDO
# ==========================
# Rejilla de barrido. Los valores rodean los de DetectParams para poder ver si el
# ajuste actual esta en un optimo local o solo es el punto de partida.
SWEEP_GRID: dict[str, tuple] = {
    "conf": (0.20, 0.25, 0.35),
    "truncated_conf": (0.35, 0.45, 0.60),
    "merge_iou": (0.45, 0.55, 0.65),
    "contain_thr": (0.75, 0.85, 0.92),
    "target_obj_frac": (0.14, 0.18, 0.24),
    "tile_overlap": (0.25, 0.35, 0.45),
}

SWEEP_MODES = ("eje", "rejilla")


def build_sweep_configs(base: D.DetectParams | None = None,
                        grid: dict[str, tuple] | None = None,
                        mode: str = "eje",
                        max_configs: int = 0) -> list[tuple[str, D.DetectParams]]:
    """Genera las configuraciones a evaluar, con su etiqueta.

    En modo 'eje' se mueve un parametro a la vez desde la base: barato en CPU y
    suficiente para ver el efecto de cada umbral. En modo 'rejilla' se recorre el
    producto completo, que captura interacciones pero cuesta el producto de todo.
    """
    base = base or D.DetectParams()
    grid = grid if grid is not None else SWEEP_GRID
    if mode not in SWEEP_MODES:
        raise ValueError(f"modo de barrido desconocido: {mode}")

    configs: list[tuple[str, D.DetectParams]] = [("base", base)]
    if mode == "eje":
        for key, values in grid.items():
            actual = getattr(base, key)
            for value in values:
                if value == actual:
                    continue
                configs.append((f"{key}={value}", base.replace(**{key: value})))
    else:
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            kw = dict(zip(keys, combo))
            label = " ".join(f"{k}={v}" for k, v in kw.items())
            if label == " ".join(f"{k}={getattr(base, k)}" for k in keys):
                continue
            configs.append((label, base.replace(**kw)))

    if max_configs and len(configs) > max_configs:
        configs = configs[:max_configs]
    return configs


@dataclass
class SweepRow:
    label: str
    report: EvaluationReport
    params: D.DetectParams

    @property
    def valid(self) -> bool:
        return self.report.valid


def run_sweep(model, annotations: AnnotationSet,
              images_dir: str | os.PathLike = TEST_IMAGES_DIR,
              configs: Sequence[tuple[str, D.DetectParams]] | None = None,
              iou_thr: float = 0.5,
              files: Sequence[str] | None = None,
              verbose: bool = True,
              predict: Callable[[D.DetectParams, Sequence[str]], PredictionSet] | None = None,
              ) -> list[SweepRow]:
    """Evalua cada configuracion reutilizando el mismo modelo ya cargado."""
    configs = list(configs) if configs is not None else build_sweep_configs()
    names = list(files) if files is not None else [a.archivo for a in annotations.imagenes]
    rows: list[SweepRow] = []
    for i, (label, params) in enumerate(configs, start=1):
        if verbose:
            print(f"[{i}/{len(configs)}] {label}", flush=True)
        preds = (predict(params, names) if predict is not None
                 else run_detector(model, names, images_dir, params))
        report = evaluate_predictions(annotations, preds.boxes, iou_thr,
                                      preds.scores, preds.seconds)
        rows.append(SweepRow(label=label, report=report, params=params))
    # Ordena por F1 y desempata por MAE de conteo: dos configuraciones con el
    # mismo acierto no son iguales si una cuenta peor.
    rows.sort(key=lambda r: (-r.report.metrics.f1, r.report.metrics.count_mae))
    return rows


def format_sweep(rows: Sequence[SweepRow], top: int = 0) -> str:
    if not rows:
        return "barrido sin configuraciones."
    invalid = any(not r.valid for r in rows)
    banner = INVALID_BANNER if invalid else VALID_BANNER
    tag = "[NO VALIDA] " if invalid else ""
    shown = list(rows[:top]) if top else list(rows)

    lines = ["=" * 100, banner, "=" * 100, "",
             f"{tag}Configuraciones evaluadas: {len(rows)}", "",
             f"  {'#':>3} {'configuracion':<34} {'F1':>7} {'prec':>7} {'rec':>7} "
             f"{'MAE':>7} {'sesgo':>7} {'s/img':>7}"]
    for i, row in enumerate(shown, start=1):
        m = row.report.metrics
        label = row.label if len(row.label) <= 34 else row.label[:31] + "..."
        lines.append(f"  {i:>3} {label:<34} {m.f1:>7.4f} {m.precision:>7.4f} "
                     f"{m.recall:>7.4f} {m.count_mae:>7.3f} {m.count_bias:>+7.3f} "
                     f"{m.seconds_per_image:>7.2f}")
    lines.append("")
    lines.append(banner)
    return "\n".join(lines)


# ==========================
# CLI
# ==========================
def _params_from_args(args) -> D.DetectParams:
    params = D.DetectParams()
    overrides = {}
    for key in ("conf", "truncated_conf", "merge_iou", "contain_thr",
                "target_obj_frac", "tile_overlap", "max_tiles"):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return params.replace(**overrides) if overrides else params


def _selected_files(names: Sequence[str], limit: int) -> list[str]:
    return list(names[:limit]) if limit and limit > 0 else list(names)


def _cmd_pre_annotate(args) -> int:
    directory = Path(args.images)
    names = _selected_files(list_images(directory), args.limit)
    if not names:
        print(f"no hay imagenes en {directory}")
        return 1

    out = Path(args.out)
    if out.is_file() and not args.force:
        previo = load_annotations(out)
        revisadas = len(previo.reviewed)
        if revisadas:
            print(f"{out} ya tiene {revisadas} imagenes revisadas a mano. "
                  "No se sobrescribe sin --force.")
            return 2

    print("PRE-ANOTACION: se generan PSEUDO-ETIQUETAS con el propio detector.")
    print("No son verdad de referencia. Sirven para corregir a mano en vez de")
    print("dibujar desde cero. Toda imagen queda con revisado=false y las")
    print("metricas medidas contra ellas no son validas como exactitud.")
    print(f"Imagenes: {len(names)}  |  pesos: {args.weights}")
    print("")

    model = load_model(args.weights)
    params = _params_from_args(args)
    data = pre_annotate(model, directory, names, params, verbose=True)

    # Si ya habia revisiones humanas y se pidio --force, no se pierden: se
    # conservan tal cual y solo se pre-anotan las imagenes nuevas.
    if out.is_file() and args.force:
        previo = load_annotations(out)
        conservadas = {a.archivo: a for a in previo.reviewed}
        if conservadas:
            data.imagenes = [conservadas.get(a.archivo, a) for a in data.imagenes]
            print(f"\nse conservan {len(conservadas)} anotaciones ya revisadas.")

    save_annotations(data, out)
    total = sum(a.conteo for a in data.imagenes)
    print("")
    print(f"escrito {out}")
    print(f"imagenes: {len(data.imagenes)}  cajas propuestas: {total}  "
          f"revisadas: {len(data.reviewed)}")
    print("Siguiente paso: corregir las cajas y poner revisado=true por imagen.")
    return 0


def _cmd_evaluate(args) -> int:
    annotations = load_annotations(args.annotations)
    names = _selected_files([a.archivo for a in annotations.imagenes], args.limit)
    if not names:
        print(f"{args.annotations} no tiene imagenes.")
        return 1

    model = load_model(args.weights)
    params = _params_from_args(args)
    preds = run_detector(model, names, args.images, params, verbose=args.verbose)
    if preds.unreadable:
        print(f"imagenes ilegibles omitidas: {', '.join(preds.unreadable)}")

    report = evaluate_predictions(annotations, preds.boxes, args.iou,
                                 preds.scores, preds.seconds)

    if not report.valid and not args.provisional:
        print(INVALID_BANNER)
        try:
            assert_reportable(report)
        except AnnotationsNotReviewed as exc:
            print(f"\nno se reportan metricas: {exc}")
        print("\nRevisa las anotaciones y marca revisado=true, o repite con "
              "--provisional para verlas marcadas como no validas.")
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_report(report, per_image=not args.no_per_image))
    return 0 if report.valid else 3


def _cmd_sweep(args) -> int:
    annotations = load_annotations(args.annotations)
    names = _selected_files([a.archivo for a in annotations.imagenes], args.limit)
    if not names:
        print(f"{args.annotations} no tiene imagenes.")
        return 1

    # La regla de validez se comprueba antes de gastar CPU: un barrido contra
    # pseudo-etiquetas solo ordenaria configuraciones por parecido al detector.
    reviewed = {a.archivo for a in annotations.reviewed}
    faltan = [n for n in names if n not in reviewed]
    if faltan and not args.provisional:
        print(INVALID_BANNER)
        print(f"\nno se ejecuta el barrido: {len(faltan)} de {len(names)} imagenes "
              "no tienen revision humana.")
        print("Marca revisado=true o repite con --provisional para un barrido "
              "explicitamente no valido.")
        return 2

    configs = build_sweep_configs(_params_from_args(args), mode=args.mode,
                                  max_configs=args.max_configs)
    print(f"barrido {args.mode}: {len(configs)} configuraciones x {len(names)} imagenes")
    model = load_model(args.weights)
    rows = run_sweep(model, annotations, args.images, configs, args.iou,
                     files=names, verbose=True)
    print("")
    print(format_sweep(rows, top=args.top))
    return 0 if rows and rows[0].valid else 3


def _cmd_status(args) -> int:
    annotations = load_annotations(args.annotations)
    total = len(annotations.imagenes)
    revisadas = len(annotations.reviewed)
    cajas = sum(a.conteo for a in annotations.imagenes)
    print(f"archivo: {args.annotations}")
    print(f"conjunto: {annotations.conjunto}  origen: {annotations.origen}")
    print(f"imagenes: {total}  revisadas: {revisadas}  sin revisar: {total - revisadas}")
    print(f"cajas anotadas: {cajas}")
    print("")
    print(VALID_BANNER if annotations.fully_reviewed else INVALID_BANNER)
    for a in annotations.unreviewed[:20]:
        print(f"  sin revisar: {a.archivo} ({a.conteo} cajas)")
    return 0 if annotations.fully_reviewed else 3


def _add_detect_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--images", default=str(TEST_IMAGES_DIR))
    parser.add_argument("--limit", type=int, default=0,
                        help="usar solo las primeras N imagenes")
    parser.add_argument("--conf", type=float)
    parser.add_argument("--truncated-conf", dest="truncated_conf", type=float)
    parser.add_argument("--merge-iou", dest="merge_iou", type=float)
    parser.add_argument("--contain-thr", dest="contain_thr", type=float)
    parser.add_argument("--target-obj-frac", dest="target_obj_frac", type=float)
    parser.add_argument("--tile-overlap", dest="tile_overlap", type=float)
    parser.add_argument("--max-tiles", dest="max_tiles", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="Arnes de evaluacion del detector de motos.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pre = sub.add_parser("pre-annotate",
                         help="pseudo-etiquetas del detector para corregir a mano")
    _add_detect_args(pre)
    pre.add_argument("--out", default=str(DEFAULT_ANNOTATIONS))
    pre.add_argument("--force", action="store_true",
                     help="sobrescribir aunque haya imagenes ya revisadas")
    pre.set_defaults(func=_cmd_pre_annotate)

    ev = sub.add_parser("evaluate", help="metricas del detector contra anotaciones")
    _add_detect_args(ev)
    ev.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    ev.add_argument("--iou", type=float, default=0.5)
    ev.add_argument("--provisional", action="store_true",
                    help="mostrar metricas sin revision humana, marcadas como no validas")
    ev.add_argument("--json", action="store_true")
    ev.add_argument("--no-per-image", action="store_true")
    ev.add_argument("--verbose", action="store_true")
    ev.set_defaults(func=_cmd_evaluate)

    sw = sub.add_parser("sweep", help="barrido de parametros ordenado por F1")
    _add_detect_args(sw)
    sw.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    sw.add_argument("--iou", type=float, default=0.5)
    sw.add_argument("--mode", choices=SWEEP_MODES, default="eje")
    sw.add_argument("--max-configs", dest="max_configs", type=int, default=0)
    sw.add_argument("--top", type=int, default=0)
    sw.add_argument("--provisional", action="store_true",
                    help="barrer sin revision humana, marcado como no valido")
    sw.set_defaults(func=_cmd_sweep)

    st = sub.add_parser("status", help="estado de revision del conjunto")
    st.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS))
    st.set_defaults(func=_cmd_status)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, AnnotationError) as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
