## 🛵 Sipark: Sistema de Monitoreo de Parqueaderos de Motos

Sipark es un sistema de monitoreo de parqueaderos de motos basado en **visión artificial**.

El sistema recibe frames de video (capturados por una cámara o simulador), detecta motocicletas utilizando **YOLO (You Only Look Once)**, asigna cada detección a zonas predefinidas (polígonos) del parqueadero y expone una API REST para que un dashboard web consuma los datos de ocupación.

### **Flujo del Sistema (Visión General)** 
1. **Cámara/Simulador** envía frames de video en tiempo real.
2. **Backend (FastAPI)** recibe cada frame como imagen a través de `/api/ingest`.
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
| **Procesamiento** | OpenCV (cv2) | Lectura de video y manipulación de frames. |
| **Geometría** | NumPy, Shapely | Operaciones numéricas y gestión de polígonos (zonas). |
| **Cliente HTTP** | requests | Utilizado en el simulador para enviar frames del video. |

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

Este componente recibe frames, procesa la visión artificial y expone la API.

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
    pip install fastapi uvicorn[standard] python-multipart numpy opencv-python shapely ultralytics requests
    ```

4.  **Iniciar Servidor:**
    ```bash
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    Backend disponible en: **`http://localhost:8000`**

> **Notas:**
> * La primera vez que se procese un frame, Ultralytics descargará el modelo **`yolo11n.pt`**.
> * El archivo **`zones.json`** debe existir en `backend/` para definir las zonas del parqueadero.

---

### 2) Simulador de Cámara (Obligatorio para Pruebas)

El simulador (`backend/simulate_camera.py`) envía frames de `test_images/video.mp4` al backend en tiempo real.

1.  **Verificar la URL del API** en `backend/simulate_camera.py`:
    ```python
    API = "[http://127.0.0.1:8000/api/ingest](http://127.0.0.1:8000/api/ingest)"
    ```

2.  **Asegurar el video de entrada:** coloca `video.mp4` en `test_images/video.mp4` (en la raíz del proyecto).

3.  **Ejecutar el Simulador** (con el backend corriendo en otra terminal):
    * Navega a la carpeta `backend/` y activa el entorno virtual (si no está activo).
    * Ejecuta:
        ```bash
        python simulate_camera.py
        ```
    * **Salida esperada (ejemplo):** `Sipark ingest: frame 000123 -> 200`

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
2.  **Ejecutar Simulador de Cámara** (envía frames del video al backend).
3.  **Iniciar Frontend** (dashboard).
4.  Abrir el Dashboard en `http://localhost:5173`.

---

## 🔗 Endpoints Principales del API

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Verifica el estado del servicio. |
| `POST` | `/api/ingest` | **Principal:** Recibe el frame para procesamiento (`form-data: file`). |
| `GET` | `/api/last` | Devuelve el último estado de ocupación (JSON). |
| `GET` | `/api/last-image` | Devuelve la última imagen procesada (con motos detectadas y zonas marcadas). |

---

## ⚠️ Solución Rápida de Problemas

| Problema | Solución |
| :--- | :--- |
| **`uvicorn` no se reconoce** | Ejecuta con el módulo de python: `python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload` |
| **`No module named fastapi`** | Instala dependencias dentro del entorno virtual: `pip install fastapi` (y las demás). |
| **Dashboard no muestra datos** | 1. Confirma que el simulador envíe frames con respuesta `200 OK`. 2. Revisa el estado de la API en `http://localhost:8000/api/last`. |
| **`/api/last-image` devuelve 404** | Aún no se ha enviado ningún frame al backend. Ejecuta el simulador. |