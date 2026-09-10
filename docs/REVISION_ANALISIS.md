# Revisión del análisis de Sipark — 2026-09-09

## Alcance

Revisión estática del backend, frontend y documentación; actualización del laboratorio de imágenes. El historial local registra cambios del laboratorio el 12 de abril de 2026, por lo que no confirma una antigüedad de nueve meses. Se conserva el modelo YOLO11m existente: cambiar pesos sin comparar resultados no demuestra una mejora.

## Cambios implementados

- El laboratorio aplica la confianza solicitada sin elevar silenciosamente los valores menores de 0.25. Se elimina además el descarte adicional que exigía 0.45 a cajas próximas al área mínima.
- Se elimina el recorte posterior a seis motos por imagen. YOLO mantiene un límite explícito de 300 candidatos; se señala cuando se alcanza.
- Se contabilizan candidatos, aceptados, descartes por regla, archivos ilegibles y duración. Los descartes anteriores a la salida de YOLO no están incluidos.
- Brillo bajo/medio/alto sustituye día/tarde/noche. Forma compacta/horizontal/vertical sustituye los supuestos ángulos. Las claves JSON históricas `angle_class`, `predominant_angle` y `angle_distribution` se conservan, pero sus valores ahora representan forma de caja; los consumidores externos deben adaptar sus categorías.
- Imágenes anotadas y exportación usan los parámetros del resultado mostrado, aunque el usuario mueva los controles después. La caché considera nombre, tamaño y fecha de modificación de las imágenes. Una desconexión no cancela el trabajo compartido.
- Excel incluye metodología, auditoría de filtros y archivos omitidos. Las detecciones se describen como aceptadas por filtros, no como verificadas.

## Hallazgos pendientes, por prioridad

1. **Conteo de producción:** `backend/main.py` recorta a dos detecciones por zona. Puede ocultar sobreocupación o errores; conviene separar conteo observado, capacidad física y alertas, una vez confirmada la capacidad real de cada zona.
2. **Evaluación real:** el repositorio contiene imágenes pero no anotaciones de referencia para medir falsos positivos, omisiones, precisión, recall o error de ocupación. Más detecciones o mayor confianza no prueban mayor exactitud.
3. **Diferencia de escenarios:** producción usa preprocesamiento y mosaicos a 1536; laboratorio usa imagen completa a 640 y filtros de tamaño. Un resultado del laboratorio no valida automáticamente el monitoreo aéreo.
4. **Filtros geométricos:** siguen activos lado mínimo de 40 px, área mínima efectiva `max(min_area, área_imagen * 0.002)`, área máxima de 65% y proporciones entre 0.38 y 2.8. Pueden excluir motos pequeñas u ocluidas; los nuevos contadores permiten investigarlo. La supresión por solapamiento también puede descartar motos reales cercanas.
5. **Servicio de producción:** la inferencia síncrona dentro de `/api/ingest` puede bloquear otras solicitudes. Imagen y resultado se publican en momentos distintos y pueden desalinearse. La ruta de zonas depende del directorio de ejecución.
6. **Instalación:** el README original omitía pandas y XlsxWriter, usados por el laboratorio. Falta un conjunto de dependencias fijado y validado para reproducir el entorno.

## Cómo evaluar una mejora del modelo

Anotar motos y ocupación por zona en imágenes representativas de altura de cámara, brillo, oclusión y densidad; separar escenas entre entrenamiento y evaluación. Comparar el sistema anterior y el candidato con los mismos datos y parámetros registrados. Medir precisión, recall, error absoluto de conteo, falsos cupos libres y latencia por escenario. Elegir umbrales según esos resultados y el costo operativo de cada error.

Referencias oficiales consultadas: [predicción de Ultralytics](https://docs.ultralytics.com/modes/predict/) y [métricas de evaluación](https://docs.ultralytics.com/guides/yolo-performance-metrics/).

## Verificación

Desde la raíz, con las dependencias del backend instaladas:

```bash
PYTHONPATH=backend backend/.venv/bin/python -m unittest discover -s backend/tests -v
```

Desde `frontend`: `npm run lint` y `npm run build`.

Las pruebas sintéticas verifican filtros, reportes y caché; no miden precisión del detector. También se ejecutó YOLO11m con una fotografía local y se generó su Excel correctamente.

Resultado de esta revisión: 7 pruebas automatizadas aprobadas, lint sin advertencias y build correcto. Las pruebas asíncronas se ejecutaron fuera del sandbox porque el cierre del executor de Python 3.14 se bloqueó también en un ejemplo mínimo independiente dentro del entorno restringido. La fotografía `WhatsApp Image 2026-04-10 at 16.52.36.jpeg` produjo 10 detecciones aceptadas y un Excel válido; este conteo no se contrastó con anotaciones humanas.

---

## Seguimiento

Los hallazgos 1 (recorte de conteo por zona), 3 (divergencia entre laboratorio y
produccion), 4 (filtros geometricos fijos), 5 (inferencia bloqueante, imagen y
resultado desalineados, ruta de zonas) y 6 (dependencias del README) se
abordaron en [el nucleo de deteccion adaptativo](DETECCION.md).

El hallazgo 2 sigue abierto y condiciona a los demas: el repositorio aun no
tiene anotaciones de referencia revisadas, asi que no hay precision ni recall
medidos. `backend/evaluation.py` es ahora la herramienta para producirlos.
