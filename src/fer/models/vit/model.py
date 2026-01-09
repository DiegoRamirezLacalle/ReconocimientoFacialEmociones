import torch
import torchvision.transforms as T
from transformers import ViTForImageClassification

# --- CONFIGURACIÓN DE LA ARQUITECTURA ---
MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"
NUM_CLASSES = 7

def get_base_model(device):
    """
    Descarga el modelo ViT experto desde HuggingFace.
    Configuración 'Limpia': Sin Dropout extra para maximizar aprendizaje rápido.
    """
    print(f"🏗️ Construyendo arquitectura ViT desde: {MODEL_NAME}")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    return model.to(device)

def get_transforms():
    """
    Devuelve las transformaciones para (Train, Validation).
    Configuración 'Limpia': Sin borrado ni blanco y negro para facilitar la tarea.
    """
    # Train: Aumentación ligera (Espejo, rotación suave y luz)
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
    ])

    # Validation/Test: Solo redimensionar y convertir a Tensor
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    return train_transform, val_transform