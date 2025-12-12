# Sipark: Sistema de Monitoreo de Parqueaderos de Motos

Sipark es un sistema de monitoreo de parqueaderos de motos basado en **visión artificial**.

El sistema funciona de la siguiente manera:
1.  El backend recibe imágenes (capturadas por una cámara o simulador).
2.  Utiliza **YOLO (You Only Look Once)** para detectar motocicletas en la imagen.
3.  Asigna cada detección a zonas predefinidas (polígonos) del parqueadero.
4.  Expone una API REST para que un dashboard web consuma los datos de ocupación.

---

## 🛠️ Tecnologías Usadas

### Backend (Visión Artificial y API)
| Componente | Tecnología | Propósito |
| :--- | :--- | :--- |
| **Lenguaje** | Python 3.10+ | Lógica principal y procesamiento de imágenes. |
| **Framework Web** | FastAPI, Uvicorn | Construcción de la API de alto rendimiento. |
| **Detección** | Ultralytics (YOLO) | Modelo de detección de objetos (motos). |
| **Procesamiento** | OpenCV (cv2) | Manipulación de imágenes. |
| **Geometría** | NumPy, Shapely | Operaciones numéricas y gestión de polígonos (zonas). |
| **Utilidades** | `python-multipart`, `CORS Middleware` | Manejo de archivos y permisos de acceso cruzado. |

### Frontend (Dashboard Web)
| Componente | Tecnología | Propósito |
| :--- | :--- | :--- |
| **Entorno** | Node.js 18+ | Entorno de ejecución y manejo de paquetes. |
| **Framework** | React | Construcción de la interfaz de usuario. |
| **Build Tool** | Vite | Empaquetador rápido para desarrollo y producción. |
| **Cliente HTTP** | Axios | Realizar peticiones al API del backend. |

---

## 🚀 Instalación y Ejecución

Asegúrate de tener instalados los **requisitos previos**: **Python 3.10+** y **Node.js 18+**.

### 1) Backend (FastAPI)

Este componente se encarga de la detección de motos y la API.

1.  **Navegar y Crear Entorno Virtual:**
    ```bash
    cd backend
    python -m venv .venv
    ```

2.  **Activar Entorno Virtual:**
    * **Windows (PowerShell):**
        ```powershell
        .\.venv\Scripts\Activate.ps1
        # Si falla por permisos, ejecuta una vez: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
        ```
    * **Linux/macOS:**
        ```bash
        source .venv/bin/activate
        ```

3.  **Instalar Dependencias:**
    ```bash
    pip install --upgrade pip
    pip install fastapi uvicorn[standard] python-multipart numpy opencv-python shapely ultralytics
    ```

4.  **Iniciar Servidor:**
    ```bash
    python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    El backend estará disponible en: **`http://localhost:8000`**

> **Nota:** La primera vez que se ejecute la detección, Ultralytics descargará automáticamente el modelo **`yolo11n.pt`**.

---

### 2) Frontend (React + Vite)

Este componente es el dashboard web que consume los datos del backend.

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
    El frontend estará disponible en: **`http://localhost:5173`**

> **Nota:** Por defecto, el frontend consume el backend en `http://localhost:8000`. Si cambias el puerto del backend, actualiza la constante `API` en `frontend/src/App.jsx`.

---

## 🔬 Pruebas y Envío de Imágenes

Para que el dashboard muestre datos, el backend debe haber recibido al menos una imagen.

### Opción A: Enviar una Imagen con `curl` (Recomendado)

Ajusta la ruta a una imagen de prueba:

```bash
curl -X POST "http://localhost:8000/api/ingest" -F "file=@ruta/a/tu_imagen.png"