import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
import io
from .inference import EmotionPredictor

app = FastAPI()

# Permitir CORS para que el frontend local funcione sin problemas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURACIÓN DE RUTAS A TUS MODELOS ---

MODEL_PATHS = {
    "cnn": "notebooks/cnn_best_exp1.pt",
    "vit": "notebooks/vit_emotions.pt" 
}

predictor = None
current_model_name = ""

# Función auxiliar para cargar modelo
def load_predictor(model_name: str):
    global predictor, current_model_name
    if model_name not in MODEL_PATHS:
        raise ValueError("Modelo desconocido")
    
    weights = MODEL_PATHS[model_name]
    predictor = EmotionPredictor(model_type=model_name, weights_path=weights)
    current_model_name = model_name

# Cargar CNN al inicio por defecto
@app.on_event("startup")
async def startup_event():
    print("🚀 Iniciando servidor...")
    try:
        load_predictor("cnn")
    except Exception as e:
        print(f"Error cargando modelo inicial: {e}")

@app.get("/")
def read_root():
    return {"status": "Online", "model": current_model_name}

@app.post("/switch-model/{model_name}")
def switch_model(model_name: str):
    if model_name not in ["cnn", "vit"]:
        raise HTTPException(status_code=400, detail="Use 'cnn' o 'vit'")
    load_predictor(model_name)
    return {"msg": f"Cambiado a {model_name}"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    emotion, prob = predictor.predict(image)
    return {
        "emotion": emotion,
        "probability": round(prob, 4),
        "model": current_model_name
    }