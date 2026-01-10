# Reconocimiento de Expresiones Faciales con CNN y Vision Transformer (ViT)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)

Sistema de reconocimiento de expresiones faciales que combina dos enfoques de aprendizaje profundo: **CNN entrenada desde cero** y **Vision Transformer (ViT) con fine-tuning**. Incluye API REST con FastAPI, interfaz web interactiva y despliegue completo con Docker.

**Autores:** Diego Ramírez & Jorge Clausen  
**Fecha:** Noviembre 2025

---

## Tabla de Contenidos

- [Características](#-características)
- [Tecnologías](#-tecnologías)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Requisitos Previos](#-requisitos-previos)
- [Instalación](#-instalación)
  - [Opción 1: Docker (Recomendado)](#opción-1-docker-recomendado)
  - [Opción 2: Instalación Local](#opción-2-instalación-local)
- [Uso](#-uso)
- [API Endpoints](#-api-endpoints)
- [Modelos](#-modelos)
- [Desarrollo](#-desarrollo)
- [Arquitectura](#-arquitectura)

---

## Características

- **Dos modelos de Deep Learning:**
  - CNN personalizada entrenada desde cero (7 clases de emociones)
  - Vision Transformer (ViT) con fine-tuning desde modelo preentrenado
- **API REST con FastAPI:**
  - Predicción en tiempo real sobre imágenes
  - Cambio dinámico entre modelos CNN/ViT
  - Retorna probabilidades para todas las emociones
- **Interfaz web interactiva:**
  - Captura desde webcam en tiempo real
  - Upload de imágenes
  - Visualización de probabilidades con gráficos
- **Contenerización completa:**
  - Docker Compose orquesta backend + frontend
  - Reproducibilidad total del entorno
  - Listo para producción

---

## Tecnologías

### Backend
- **Python 3.10**
- **PyTorch 2.1** (CUDA 12.1)
- **FastAPI** para API REST
- **Uvicorn** como servidor ASGI
- **Transformers** (HuggingFace) para ViT
- **timm** para modelos de visión
- **OpenCV**, **MediaPipe**, **MTCNN** para preprocesamiento

### Frontend
- HTML5, CSS3, JavaScript vanilla
- WebRTC para captura de webcam

### DevOps
- **Docker** y **Docker Compose**
- **Nginx** como proxy reverso y servidor estático

---

## Estructura del Proyecto

```
ReconocimientoFacialEmociones/
├── src/
│   ├── api/                    # API (estructura alternativa, vacía)
│   ├── backen/                 # Backend principal FastAPI
│   │   ├── main.py            # Entrypoint de la API
│   │   ├── inference.py       # Clase EmotionPredictor
│   │   └── index.html         # Frontend integrado
│   └── fer/                    # Módulo de Deep Learning
│       ├── models/            # Arquitecturas CNN/ViT
│       ├── train/             # Scripts de entrenamiento + checkpoints (.pt)
│       ├── eval/              # Evaluación y métricas
│       └── utils/             # Utilidades
├── data/
│   └── raw/                   # Datasets (FER-2013, AffectNet)
│       ├── train/
│       ├── val/
│       └── test/
├── configs/                   # Configuraciones YAML
├── docker/
│   ├── docker-compose.yml     # Orquestación de servicios
│   └── nginx.conf             # Config Nginx
├── notebooks/                 # Jupyter notebooks de exploración
├── Dockerfile                 # Imagen Docker del backend
├── pyproject.toml             # Dependencias del proyecto
└── README.md                  # Este archivo
```

---

---

## Organización de carpetas

El proyecto sigue una estructura modular que separa claramente **backend**, **modelos**, **datos**, **entrenamiento** y **despliegue**.

###  `src/backen/`
Contiene la aplicación principal de FastAPI:
- `main.py`: Entrypoint de la API REST
- `inference.py`: Lógica de carga de modelos y predicción
- `index.html`: Interfaz web integrada (frontend ligero)

---

###  `src/fer/`
Módulo de Deep Learning (Facial Emotion Recognition):

#### 🔸 `models/`
- Definición de arquitecturas CNN y ViT
- Clases PyTorch (`nn.Module`)

#### 🔸 `train/`
- Scripts de entrenamiento
- Checkpoints finales de los modelos (`.pt`)
- **Estos pesos son los que se cargan en producción**

#### 🔸 `eval/`
- Evaluación de modelos
- Métricas (accuracy, confusion matrix, etc.)

#### 🔸 `utils/`
- Funciones auxiliares de preprocesado
- Normalización, transformaciones y helpers

---

###  `data/`
Contiene los datasets **solo para entrenamiento** (no incluidos en Docker):

```text
data/
└── raw/
    ├── fer2013/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── affectnet/
        ├── train/
        ├── val/
        └── test/


##  Requisitos Previos

### Para Docker (Recomendado)
- [Docker](https://www.docker.com/get-started) >= 20.10
- [Docker Compose](https://docs.docker.com/compose/install/) >= 2.0
- (Opcional) GPU NVIDIA + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) para inferencia con CUDA

### Para Instalación Local
- Python 3.10+
- CUDA 12.1 (opcional, para GPU)
- Git

---

##  Instalación

### Opción 1: Docker (Recomendado)

**1. Clonar el repositorio:**
```bash
git clone https://github.com/tu-usuario/ReconocimientoFacialEmociones.git
cd ReconocimientoFacialEmociones
```

**2. Construir y lanzar los servicios:**
```bash
cd docker
docker compose up --build
```

**3. Acceder a la aplicación:**
- **Frontend:** http://localhost
- **API (docs):** http://localhost/api/docs
- **API directa:** http://localhost:8000

**Para detener:**
```bash
docker compose down
```

---

### Opción 2: Instalación Local

**1. Clonar el repositorio:**
```bash
git clone https://github.com/tu-usuario/ReconocimientoFacialEmociones.git
cd ReconocimientoFacialEmociones
```

**2. Crear entorno virtual:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

**3. Instalar PyTorch con CUDA (o CPU):**
```bash
# CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0

# O solo CPU
pip install torch==2.1.0 torchvision==0.16.0
```

**4. Instalar dependencias:**
```bash
pip install -e .
```

**5. Ejecutar el servidor:**
```bash
uvicorn src.backen.main:app --host 0.0.0.0 --port 8000 --reload
```

**6. Abrir el frontend:**
- Navega a http://localhost:8000
- O abre directamente `src/backen/index.html` en el navegador

---

##  Uso

### Interfaz Web

1. **Webcam en tiempo real:**
   - Click en "Activar Cámara"
   - Presiona "Capturar y Predecir"
   - Ve las probabilidades de cada emoción en tiempo real

2. **Subir imagen:**
   - Click en "Subir Imagen"
   - Selecciona una foto con rostro
   - Ve la predicción instantánea

3. **Cambiar modelo:**
   - Botones "CNN" / "ViT" para alternar entre modelos
   - Observa diferencias en predicciones

### Línea de Comandos (curl)

```bash
# Predecir emoción en una imagen
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"

# Cambiar a modelo ViT
curl -X POST "http://localhost:8000/switch-model/vit"

# Cambiar a modelo CNN
curl -X POST "http://localhost:8000/switch-model/cnn"
```

### Python

```python
import requests

# Predecir emoción
with open("imagen.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    print(response.json())
    # {
    #   "dominant_emotion": "Happy",
    #   "predictions": {
    #     "Happy": 0.85,
    #     "Neutral": 0.10,
    #     "Surprise": 0.03,
    #     ...
    #   },
    #   "model": "cnn"
    # }
```

---

##  API Endpoints

### `GET /`
Retorna la interfaz web HTML.

**Response:** HTML frontend

---

### `POST /predict`
Predice la emoción en una imagen.

**Request:**
- `Content-Type: multipart/form-data`
- `file`: Archivo de imagen (JPG, PNG, etc.)

**Response:**
```json
{
  "dominant_emotion": "Happy",
  "predictions": {
    "Angry": 0.02,
    "Disgust": 0.01,
    "Fear": 0.01,
    "Happy": 0.85,
    "Sad": 0.03,
    "Surprise": 0.05,
    "Neutral": 0.03
  },
  "model": "cnn"
}
```

---

### `POST /switch-model/{model_name}`
Cambia el modelo activo.

**Path Parameters:**
- `model_name`: `"cnn"` o `"vit"`

**Response:**
```json
{
  "msg": "Cambiado a vit",
  "model": "vit"
}
```

---

##  Modelos

### CNN Personalizada
- **Arquitectura:** 4 bloques Conv + BatchNorm + ReLU + MaxPool
- **Parámetros:** ~68M (68MB checkpoint)
- **Input:** Imágenes en escala de grises 224×224
- **Output:** 7 clases de emociones

**Clases:**
1. Angry (Enfado)
2. Disgust (Asco)
3. Fear (Miedo)
4. Happy (Feliz)
5. Sad (Triste)
6. Surprise (Sorpresa)
7. Neutral (Neutral)

### Vision Transformer (ViT)
- **Base:** `mo-thecreator/vit-Facial-Expression-Recognition`
- **Fine-tuning:** Adaptado a 7 clases con dataset combinado
- **Parámetros:** ~343M (343MB checkpoint)
- **Input:** Imágenes RGB 224×224
- **Output:** 7 clases de emociones

**Ventajas ViT vs CNN:**
- Mayor precisión en escenarios complejos
- Mejor generalización
- Atención global sobre la imagen

---
---

##  Datasets utilizados

Para el entrenamiento y evaluación de los modelos de reconocimiento de emociones se han utilizado **datasets públicos ampliamente empleados en investigación**, descargados directamente de sus fuentes oficiales.

### 🔹 FER-2013
- **Descripción:** Dataset clásico para reconocimiento de expresiones faciales en escala de grises.
- **Número de clases:** 7 emociones
- **Resolución:** 48×48 píxeles (re-escaladas a 224×224 durante el preprocesado)
- **Formato:** Imágenes en carpetas por clase

 **Fuente oficial:**
- Kaggle – FER-2013  
  https://www.kaggle.com/datasets/msambare/fer2013

 **Licencia:** Uso académico e investigación.

---

### 🔹 AffectNet
- **Descripción:** Dataset a gran escala de expresiones faciales anotadas manualmente.
- **Número de clases:** 7 emociones básicas
- **Resolución:** Variable (normalizadas a 224×224)
- **Formato:** Imágenes RGB

**Fuente oficial:**
- Sitio web oficial de AffectNet  
  http://mohammadmahoor.com/affectnet/


---

### 🔹 Uso de los datasets en el proyecto
- Los datasets se utilizan **exclusivamente en la fase de entrenamiento y evaluación**
- **No se incluyen dentro del contenedor Docker**
- Durante el despliegue solo se cargan los **pesos entrenados (`.pt`)**

##  Desarrollo

### Ejecutar tests
```bash
pytest tests/ -v
```

### Entrenar modelos

**CNN:**
```bash
python -m fer.train.train_cnn --config configs/train_cnn.yaml
```

**ViT:**
```bash
python -m fer.train.train_vit --config configs/train_vit.yaml
```

### Hot reload en desarrollo
```bash
uvicorn src.backen.main:app --reload --host 0.0.0.0 --port 8000
```

---

##  Arquitectura

```
┌─────────────────┐
│   Frontend Web  │  (HTML5 + JS + WebRTC)
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   Nginx Proxy   │  (Puerto 80, opcional)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Backend│  (Puerto 8000)
│   (Uvicorn)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ EmotionPredictor│
│   (PyTorch)     │
│                 │
│  ┌──────────┐   │
│  │   CNN    │   │  (68MB)
│  └──────────┘   │
│                 │
│  ┌──────────┐   │
│  │   ViT    │   │  (343MB)
│  └──────────┘   │
└─────────────────┘
```

---


##  Autores

- **Diego Ramírez Lacalle** - [GitHub](https://github.com/DiegoRamirezLacalle)
- **Jorge Clausen** - [GitHub](https://github.com/jorge-clausen)

**Universidad de Deusto** - Proyecto Reconocimiento Facial de Emociones  
Noviembre 2025

---



