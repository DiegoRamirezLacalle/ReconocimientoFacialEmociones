import torch
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"
NUM_CLASSES = 7


def create_vit(device: str | None = None):
    """
    Crea el ViT preentrenado y el processor.
    Devuelve (model, processor, device).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

    model.to(device)
    print(f"Modelo ViT cargado correctamente en {device}")
    return model, processor, device
