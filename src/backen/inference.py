import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# Etiquetas estándar (FER-2013 / AffectNet subset)
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- 1. ARQUITECTURA CNN (Idéntica a tu entrenamiento) ---
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7, in_channels=1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# --- 2. CLASE GESTORA DE MODELOS ---
class EmotionPredictor:
    def __init__(self, model_type="cnn", weights_path=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        print(f"🔄 Cargando modelo: {model_type} en {self.device}...")

        if model_type == "cnn":
            self.model = EmotionCNN(num_classes=7, in_channels=1)
            # Transformación para CNN: Gris + Resize
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        
        elif model_type == "vit":
            # Cargamos la arquitectura base igual que en tu script de entrenamiento
            model_name_hf = "mo-thecreator/vit-Facial-Expression-Recognition"
            self.processor = ViTImageProcessor.from_pretrained(model_name_hf)
            self.model = ViTForImageClassification.from_pretrained(
                model_name_hf, 
                num_labels=7, 
                ignore_mismatched_sizes=True
            )

        # Cargar los pesos personalizados (.pt)
        if weights_path:
            try:
                print(f"📂 Leyendo pesos desde: {weights_path}")
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print("✅ Pesos cargados correctamente.")
            except Exception as e:
                print(f"❌ Error cargando pesos: {e}")
                print("⚠️ Usando pesos aleatorios/base (cuidado, la predicción será mala).")

        self.model.to(self.device)
        self.model.eval()
# En inference.py inside EmotionPredictor class

    def predict(self, image):
        # Preprocesamiento
        if self.model_type == "cnn":
            inputs = self.transform(image).unsqueeze(0).to(self.device)
        else:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Inferencia
        with torch.no_grad():
            if self.model_type == "cnn":
                logits = self.model(inputs)
            else:
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Aplicar Softmax para obtener probabilidades (0 a 1)
            probs = torch.nn.functional.softmax(logits, dim=1)
            
            # Convertir a lista de Python
            probs_list = probs[0].tolist()
            
            # Crear diccionario { "Happy": 0.85, "Sad": 0.02, ... }
            result = {label: score for label, score in zip(EMOTION_LABELS, probs_list)}
            
            return result