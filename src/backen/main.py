import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from .inference import EmotionPredictor  #

app = FastAPI()

# Permitir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURACIÓN DE RUTAS ---
CNN_WEIGHTS = os.getenv("CNN_WEIGHTS", "src/fer/train/cnn_best_exp1.pt")
VIT_WEIGHTS = os.getenv("VIT_WEIGHTS", "src/fer/train/vit_emotions.pt")


MODEL_PATHS = {
    "resnet": CNN_WEIGHTS,
    "vit": VIT_WEIGHTS 
}

predictor = None
current_model_name = ""

def load_predictor(model_name: str):
    global predictor, current_model_name
    
    # Validación de seguridad
    if model_name not in MODEL_PATHS:

        raise ValueError(f"Modelo '{model_name}' no reconocido. Opciones: {list(MODEL_PATHS.keys())}")
    
    weights = MODEL_PATHS[model_name]
    
    try:
        
        predictor = EmotionPredictor(model_type=model_name, weights_path=weights)
        current_model_name = model_name
        print(f" Modelo {model_name} cargado correctamente desde {weights}")
    except Exception as e:
        print(f" Error fatal al cargar el predictor: {e}")


@app.on_event("startup")
async def startup_event():
    print(" Iniciando servidor...")
    try:
        load_predictor("resnet")
    except Exception as e:
        print(f"Error cargando modelo inicial: {e}")

@app.get("/")
def read_root():
    return {"status": "Online", "model": current_model_name}

@app.post("/switch-model/{model_name}")
def switch_model(model_name: str):
    if model_name not in MODEL_PATHS.keys():
        raise HTTPException(status_code=400, detail="Use 'resnet' o 'vit'")
    
    load_predictor(model_name)
    return {"msg": f"Cambiado a {model_name}"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not predictor:
        raise HTTPException(status_code=500, detail="Modelo no cargado. Revisa los logs del servidor.")
    
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        predictions = predictor.predict(image)
        dominant_emotion = max(predictions, key=predictions.get)
        
        return {
            "dominant_emotion": dominant_emotion,
            "predictions": predictions,
            "model": current_model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")