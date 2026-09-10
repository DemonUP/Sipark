# Nucleo de deteccion adaptativo — 2026-09-09

Continua la [revision del analisis](REVISION_ANALISIS.md). Aquella dejo seis
hallazgos pendientes; este trabajo aborda los que se podian resolver sin
anotaciones humanas y deja medible lo que no.

## Punto de partida

El diagnostico encontro un desajuste de fondo. `zones.json` describia una
rejilla de 24 celdas sobre 1024x1024, `main.py` aplicaba un preproceso llamado
`preprocess_drone` y recortaba a dos motos por zona: todo el servicio estaba
calibrado para una vista aerea superior. Las 19 fotografias de camara del
repositorio son oblicuas, a nivel de suelo, con mas de cien motos y oclusion
severa. La unica imagen superior, `img1.png`, esta generada por computador y
lleva su marca de agua.

Ademas, bajar el tamano de mosaico subia el conteo de 14 a 55 en una fotografia
oblicua, pero la imagen anotada mostraba que esas cajas de mas eran duplicados y
partes de moto: una rueda, media moto, recortes en el borde del mosaico. Mas
detecciones no era mas exactitud.

## Que se hizo

### Un nucleo comun que mide la escena

`backend/detection.py` sustituye las dos rutas de inferencia que existian —una
en el servicio, otra en el laboratorio— por una sola, y no asume el montaje de
la camara: lo estima en cada imagen.

Una pasada sobre la imagen completa y cuatro cuadrantes miden el tamano de las
motos y como cambia con la altura de la imagen. Sobre las medianas por franja se
ajusta una recta. Si el tamano se mantiene, la vista es superior; si crece con
la profundidad, es oblicua. Con esa pendiente se elige, por franja, el tamano de
mosaico con el que las motos ocupan una fraccion util del recorte: recortes
pequenos donde se ven pequenas, amplios donde se ven grandes.

El ajuste se declara fiable solo si las motos abarcan profundidad suficiente.
Apinadas en una sola franja la pendiente no es medible, y darla por nula
equivaldria a afirmar que la vista es superior. En ese caso la vista queda
`indeterminado`, el filtro de escala se desactiva y el plan recorre varias
escalas en lugar de apostar por una.

### Cuatro reglas de fusion

El problema no era detectar poco, era distinguir una moto de una vista parcial
de otra. Sobre los candidatos de todos los mosaicos se aplica:

- **Supresion por solape** (NMS), con la caja cortada por el borde de su mosaico
  penalizada: es una vista parcial, asi que no debe desplazar a la caja que si
  abarca el objeto completo, aunque el detector le asigne mas confianza.
- **Parte de otra moto**: se descarta la caja muy incluida en otra *y*
  sensiblemente mas pequena. Exigir las dos condiciones evita borrar dos motos
  vecinas que caen dentro de una caja grande, frecuente en filas apinadas.
- **Caja envolvente**: el caso simetrico, una caja que no aporta un objeto nuevo
  sino que engloba a varios ya aceptados. El area de la union se calcula de
  forma exacta; sumar los solapes por separado la sobreestimaria y descartaria
  motos reales.
- **Coherencia de escala** frente al tamano esperado a esa profundidad, y solo
  con la geometria medida. Sin medir, el tamano de referencia vendria del primer
  plano y castigaria justamente a las motos lejanas.

### Correcciones en el servicio

- Se elimino el recorte a dos detecciones por zona. Ocultaba justamente la
  sobreocupacion y los errores que hay que poder ver. Ahora se informan el
  conteo observado, la capacidad declarada de cada zona y las zonas por encima
  de su capacidad, por separado.
- La inferencia salio del hilo del servidor. Antes bloqueaba la API entera
  mientras procesaba una imagen.
- Imagen y resultado se publican juntos. Por separado, un cliente podia leer una
  imagen que no correspondia al estado mostrado.
- `zones.json` se lee junto al modulo, no en el directorio de ejecucion.
- Desaparecieron los umbrales aplicados en silencio: la confianza pedida es la
  usada, y `DEFAULT_CONF=0.012` elevado a `MIN_SCORE=0.10` ya no existe.

### Aviso cuando las zonas no describen cupos

`zones.json` declara ahora para que vista se dibujaron sus poligonos. Si la
geometria medida no coincide, el servicio avisa y el dashboard muestra `n/d` en
los cupos en vez de un numero firme. No basta con contar cuantas motos quedan
fuera de las zonas: con la rejilla extendida sobre toda la imagen, en una foto
oblicua el 96 % de las cajas cae dentro de alguna celda por casualidad, y la
ocupacion resultante no significa nada.

### Geometria reutilizada, desactivada por defecto

En una camara fija el montaje no cambia entre fotogramas, asi que la geometria
puede reaprovecharse: ahorra cuatro inferencias por imagen y baja el proceso de
8,1 s a 4,7 s, con resultado identico sobre la misma fotografia.

Viene desactivada (`SIPARK_SCENE_REFRESH=0`, medir siempre) porque atar el plan
de mosaicos a la geometria de un fotograma anterior cuesta detecciones cuando la
escena no es exactamente la misma: en una prueba con dos encuadres distintos del
mismo parqueadero, 25 motos en vez de 28. Con una camara realmente fija ese
costo desaparece.

La reutilizacion no es ciega. La pasada sobre la imagen completa se ejecuta
siempre, asi que sirve de contraste sin costo: si el tamano observado no
concuerda con el que la geometria guardada predice, se descarta y se vuelve a
medir. Un cambio de tamano de imagen la invalida de inmediato. Sin esa
comprobacion, una fotografia aerea procesada tras una oblicua daba 10 motos en
vez de 22, y el error persistia durante muchos fotogramas: lo encontro la prueba
en vivo contra el servidor, no la suite.

## Que se puede afirmar y que no

En `img1.png` el resultado es exacto: **22 detecciones sobre 22 motos**, sin
falsos positivos ni omisiones, con confianza minima de 0,77 — incluido el cupo
que tiene dos motos. Es la unica imagen del repositorio cuyo conteo se puede
establecer a ojo con certeza, y sirvio para encontrar la regla que faltaba: una
caja de confianza 0,25 que englobaba dos motos reales y que ni la supresion por
solape ni la regla de partes atrapaban.

En las fotografias oblicuas **no se puede afirmar una mejora de exactitud**. El
conteo pasa de 14 a 17 en una de ellas, pero sin anotaciones humanas ese numero
no distingue una moto recuperada de un falso positivo. El detector es
determinista —tres corridas dan cajas identicas—, asi que la medicion es
posible; lo que falta son las anotaciones.

Las filas densas del fondo siguen en su mayoria sin detectar. Es el techo de un
modelo entrenado en COCO sobre una escena de mas de cien motos ocluidas, y
ninguna combinacion de parametros lo levanta.

## Como medir una mejora

`backend/evaluation.py` es el arnes que faltaba:

```bash
cd backend
# 1) propone cajas para no anotar desde cero (pseudo-etiquetas)
PYTHONPATH=. .venv/bin/python -m evaluation pre-annotate
# 2) corregir a mano annotations/test_images.json y poner revisado=true
# 3) precision, recall, F1, MAE de conteo
PYTHONPATH=. .venv/bin/python -m evaluation evaluate
# 4) barrido de parametros ordenado por F1
PYTHONPATH=. .venv/bin/python -m evaluation sweep
```

`evaluate` y `sweep` **se niegan a reportar metricas** mientras las anotaciones
no esten revisadas por una persona. Medir contra las pseudo-etiquetas del propio
detector seria compararlo consigo mismo: da cifras altas y no dice nada.

## Costo y perfiles

Solo CPU, sin GPU, con `yolo11m`. Medido por imagen:

| Perfil | Vista oblicua | Vista superior |
| :--- | :--- | :--- |
| `max_tiles=60` (por defecto) | 25 mosaicos, 17 motos, 8,3 s | 4 mosaicos, 22 motos, 3,2 s |
| `max_tiles=12` | 11 mosaicos, 16 motos, 4,0 s | igual, 3,2 s |
| `max_tiles=6` | 6 mosaicos, 14 motos, 3,1 s | igual, 3,2 s |

En vista superior el presupuesto no cambia nada: la escena ya se resuelve con
cuatro mosaicos. El dashboard consulta cada 1,5 s, por debajo del tiempo de
proceso; para un feed en vivo conviene `SIPARK_MAX_TILES=12`, un modelo mas
liviano (`yolo11s`, `yolo11n`) o una GPU.

## Pendiente, por prioridad

1. **Anotar.** Sin anotaciones revisadas no hay exactitud medida, solo conteos.
   Es el requisito de todo lo demas.
2. **Definir el montaje real y redibujar las zonas.** Si la camara sera oblicua,
   la rejilla de 24 celdas sobre 1024x1024 no delimita cupos y hay que trazar los
   poligonos sobre el encuadre real. Si sera aerea, hacen falta fotografias
   aereas reales: `img1.png` es sintetica.
3. **Declarar la capacidad real de cada zona** con `capacity` en `zones.json`. El
   valor por defecto es 1, y con el `img1.png` informa `E9` por encima de su
   capacidad, que es correcto para un cupo de una moto.
4. **Entrenar con imagenes locales anotadas.** Es el unico camino para las filas
   densas del fondo. Requiere volumen de anotacion y horas de computo.
5. **Latencia**, si el objetivo es tiempo real: modelo mas liviano o GPU.

Referencias: [prediccion de Ultralytics](https://docs.ultralytics.com/modes/predict/),
[metricas de evaluacion](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

## Verificacion

```bash
PYTHONPATH=backend backend/.venv/bin/python -m unittest discover -s backend/tests
cd frontend && npm run lint && npm run build
```

Resultado de esta ronda: 76 pruebas automatizadas aprobadas, lint sin
advertencias, build correcto, e ingesta comprobada de punta a punta sobre
imagenes reales. Las pruebas cubren la geometria y las reglas de fusion con
cajas sinteticas; no miden la exactitud del detector.
