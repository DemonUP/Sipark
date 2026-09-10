## 🛵 Sipark: Sistema de Monitoreo de Parqueaderos de Motos

Sipark es un sistema de monitoreo de parqueaderos de motos basado en **visión artificial**.

El sistema recibe imágenes (capturadas por una cámara o simulador), detecta motocicletas utilizando **YOLO (You Only Look Once)**, asigna cada detección a zonas predefinidas (polígonos) del parqueadero y expone una API REST para que un dashboard web consuma los datos de ocupación.

### **Flujo del Sistema (Visión General)** 
1. **Cámara/Simulador** envía una imagen periódicamente.
2. **Backend (FastAPI)** recibe la imagen a través de `/api/ingest`.
3. **YOLO** detecta motos y **Shapely** asigna las detecciones a las zonas.
4. El backend expone el estado de ocupación a través de `/api/last`.
5. **Frontend (React)** consume `/api/last` y muestra el dashboard.

---

## 🛠️ Tecnologías Usadas

### Backend (Visión Artificial y API)
| Componente | Tecnología | Propósito |
| :--- | :--- | :--- |
| **Lenguaje** | Python 3.10+ | Lógica principal y procesamiento de imágenes. |
| **Framework Web** | FastAPI, Uvicorn | Construcción de la API de alto rendimiento. |
| **Detección** | Ultralytics (**YOLO**) | Modelo de detección de objetos (motos). |
| **Procesamiento** | OpenCV (cv2) | Manipulación de imágenes. |
| **Geometría** | NumPy, Shapely | Operaciones numéricas y gestión de polígonos (zonas). |
| **Cliente HTTP** | requests | Utilizado en el simulador para enviar imágenes. |

### Frontend (Dashboard Web)
| Componente | Tecnología | Propósito |
| :--- | :--- | :--- |
| **Entorno** | Node.js 18+ | Entorno de ejecución. |
| **Framework** | React, Vite | Construcción de la interfaz de usuario. |
| **Cliente HTTP** | Axios | Realizar peticiones al API del backend. |

---

## 🚀 Instalación y Ejecución

Asegúrate de cumplir con los **Requisitos previos**: **Python 3.10+** y **Node.js 18+**.

### 1) Backend (FastAPI)

Este componente recibe las imágenes, procesa la visión artificial y expone la API.

1.  **Navegar y Crear Entorno Virtual:**
    ```bash
    cd backend
    python -m venv .venv
    ```

2.  **Activar Entorno Virtual:**
    * **Windows (PowerShell):**
        ```powershell
        .\.venv\Scripts\Activate.ps1
        # Si PowerShell bloquea, ejecuta una vez: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
        ```
    * **Linux/macOS:**
        ```bash
        source .venv/bin/activate
        ```

3.  **Instalar Dependencias:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    > En una máquina sin GPU, instala antes `torch` de CPU para no descargar las bibliotecas de CUDA:
    > `pip install torch --index-url https://download.pytorch.org/whl/cpu`

4.  **Iniciar Servidor:**
    ```bash
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    Backend disponible en: **`http://localhost:8000`**

> **Notas:**
> * La primera vez que se procese una imagen, Ultralytics descargará el modelo **`yolo11m.pt`**.
> * El archivo **`zones.json`** debe existir en `backend/` para definir las zonas del parqueadero.

---

### 2) Simulador de Cámara (Obligatorio para Pruebas)

El simulador (`backend/simulate_camera.py`) envía imágenes al backend de forma periódica.

1.  **Verificar la URL del API** en `backend/simulate_camera.py`:
    ```python
    API = "[http://127.0.0.1:8000/api/ingest](http://127.0.0.1:8000/api/ingest)"
    ```

2.  **Ejecutar el Simulador** (con el backend corriendo en otra terminal):
    * Navega a la carpeta `backend/` y activa el entorno virtual (si no está activo).
    * Ejecuta:
        ```bash
        python simulate_camera.py
        ```
    * **Salida esperada (ejemplo):** `Sipark ingest: img1.png -> 200`

> **Solución de problemas del simulador:**
> * Si aparece `Connection refused`: El backend no está corriendo en `http://127.0.0.1:8000`.
> * Si aparece `requests has no attribute post`: Revisa que no exista un archivo llamado `requests.py` en `backend/` que esté sobrescribiendo la librería real.

---

### 3) Frontend (React + Vite)

Este componente es el dashboard web.

1.  **Abrir otra terminal y Navegar:**
    ```bash
    cd frontend
    ```

2.  **Instalar Dependencias:**
    ```bash
    npm install
    ```

3.  **Iniciar Servidor de Desarrollo:**
    ```bash
    npm run dev
    ```
    Frontend disponible en: **`http://localhost:5173`**

> **Nota:** Si cambias el puerto del backend, actualiza la constante `API` en `frontend/src/App.jsx`.

---

## 🔁 Flujo Recomendado para Correr Sipark

1.  **Iniciar Backend** (puerto 8000).
2.  **Ejecutar Simulador de Cámara** (envía imágenes al backend).
3.  **Iniciar Frontend** (dashboard).
4.  Abrir el Dashboard en `http://localhost:5173`.

---

## 🔗 Endpoints Principales del API

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Verifica el estado del servicio. |
| `POST` | `/api/ingest` | **Principal:** Recibe la imagen para el procesamiento (`form-data: file`). |
| `GET` | `/api/last` | Devuelve el último estado de ocupación (JSON). |
| `GET` | `/api/last-image` | Devuelve la última imagen procesada (con motos detectadas y zonas marcadas). |

---

## ⚠️ Solución Rápida de Problemas

| Problema | Solución |
| :--- | :--- |
| **`uvicorn` no se reconoce** | Ejecuta con el módulo de python: `python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload` |
| **`No module named fastapi`** | Instala dependencias dentro del entorno virtual: `pip install fastapi` (y las demás). |
| **Dashboard no muestra datos** | 1. Confirma que el simulador envíe imágenes con respuesta `200 OK`. 2. Revisa el estado de la API en `http://localhost:8000/api/last`. |
| **`/api/last-image` devuelve 404** | Aún no se ha enviado ninguna imagen al backend. Ejecuta el simulador. |

---

## 🎯 Detección de Motos

La detección vive en `backend/detection.py` y la comparten el monitoreo y el laboratorio. No asume el montaje de la cámara: en cada imagen mide el tamaño de las motos y cómo cambia con la profundidad, clasifica la vista como **aérea** u **oblicua**, y con esa medida elige por franjas el tamaño de mosaico. Sobre los candidatos aplica cuatro reglas de fusión que descartan duplicados, partes de moto, cajas que engloban a varias y tamaños incoherentes con la escena.

El resultado de `/api/last` incluye la geometría detectada en `scene` y la auditoría de descartes en `diagnostics`.

### Variables de entorno

| Variable | Por defecto | Propósito |
| :--- | :--- | :--- |
| `SIPARK_MODEL` | `yolo11m.pt` | Pesos. `yolo11s`/`yolo11n` son más rápidos en CPU. |
| `SIPARK_CONF` | `0.25` | Confianza mínima. Se usa tal cual, sin elevarla en silencio. |
| `SIPARK_TRUNCATED_CONF` | `0.45` | Confianza exigida a una caja cortada por el borde de su mosaico. |
| `SIPARK_MAX_TILES` | `60` | Presupuesto de mosaicos. Bájelo a `12` para un feed en vivo. |
| `SIPARK_SCENE_REFRESH` | `0` | Fotogramas que reaprovechan la geometría medida. `0` la mide en cada imagen. Con una cámara fija, `20` baja el proceso de 8,1 s a 4,7 s. |
| `SIPARK_ZONE_CAPACITY` | `1` | Cupos por zona, si `zones.json` no los declara. |
| `SIPARK_PREPROCESS` | `0` | Realce de contraste. Útil con poca luz. |

> **Nota:** solo CPU, con `yolo11m`, el proceso tarda entre 3 y 8 segundos por imagen. El dashboard consulta cada 1,5 s, así que para tiempo real hace falta un modelo más liviano o una GPU.

### Zonas y cupos

`zones.json` declara con `"view"` para qué vista se dibujaron sus polígonos, y admite `"capacity"` por zona. Si la geometría medida no coincide con la declarada, el servicio avisa y el dashboard muestra `n/d` en los cupos: un polígono trazado sobre una vista aérea no delimita un cupo en una vista oblicua. El conteo de motos detectadas se informa siempre.

Las zonas por encima de su capacidad se reportan en `totals.spaces_over_capacity` y `zones_over_capacity`. El conteo nunca se recorta al cupo: eso ocultaría la sobreocupación.

---

## 📊 Evaluación y Análisis

`backend/evaluation.py` mide exactitud contra anotaciones de referencia:

```bash
cd backend
PYTHONPATH=. .venv/bin/python -m evaluation pre-annotate   # propone cajas
# corregir annotations/test_images.json y poner revisado=true
PYTHONPATH=. .venv/bin/python -m evaluation evaluate       # precision, recall, F1, MAE
PYTHONPATH=. .venv/bin/python -m evaluation sweep          # barrido de parámetros
```

`evaluate` y `sweep` se niegan a reportar métricas mientras las anotaciones no estén revisadas por una persona: medir contra las pseudo-etiquetas del propio detector sería compararlo consigo mismo.

El laboratorio incluye auditoría de descartes, la geometría medida por imagen y exportación a Excel. Consulta [el núcleo de detección y sus límites](docs/DETECCION.md) y [la revisión previa](docs/REVISION_ANALISIS.md).

### Pruebas

```bash
PYTHONPATH=backend backend/.venv/bin/python -m unittest discover -s backend/tests
```
