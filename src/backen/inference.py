import torch
import torch.nn as nn
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

from src.fer.models.cnn.model import EmotionResNetSmall 

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class EmotionPredictor:
    def __init__(self, model_type="resnet", weights_path=None, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        print(f"Cargando modelo: {model_type} en {self.device}...")

        if model_type == "resnet":
            self.model = EmotionResNetSmall(num_classes=7, in_channels=1)
            
            # Transformación correcta para ResNet (Gris -> 224x224 -> Tensor)
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        
        elif model_type == "vit":
            model_name_hf = "mo-thecreator/vit-Facial-Expression-Recognition"
            self.processor = ViTImageProcessor.from_pretrained(model_name_hf)
            self.model = ViTForImageClassification.from_pretrained(
                model_name_hf, 
                num_labels=7, 
                ignore_mismatched_sizes=True
            )

        # Cargar los pesos (.pt)
        if weights_path:
            self._load_weights(weights_path)
        else:
            print("No se han proporcionado pesos. ")

        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Pesos cargados correctamente.")
        except RuntimeError as e:
            print(f" Error en Runtime: {e}")
        except Exception as e:
            print(f" Error, ha saltado la excepción: {e}")

    def predict(self, image):
        # Preprocesamiento
        if self.model_type == "resnet":
            # ResNet espera [Batch, 1, 224, 224]
            inputs = self.transform(image).unsqueeze(0).to(self.device)
        else:
            # ViT usa su procesador propio
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Inferencia
        with torch.no_grad():
            if self.model_type == "resnet":
                logits = self.model(inputs)
            else:
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Softmax
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs_list = probs[0].tolist()
            
            result = {label: score for label, score in zip(EMOTION_LABELS, probs_list)}
            return result