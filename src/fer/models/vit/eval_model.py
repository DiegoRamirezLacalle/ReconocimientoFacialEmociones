import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm

# --- CONFIG ---
MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"
NUM_CLASSES = 7
MODEL_OUT = r"C:\Users\joorg\Documents\ReconocimientoFacialEmociones\notebooks\vit_emotions.pt"
DATA_ROOT =  "data"
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate():
    print(f"Evaluando en dispositivo: {device}")

    # CARGAR MODELO PREENTRENADO + PESOS ENTRENADOS
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    ).to(device)

    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    model.eval()

    print("Modelo cargado desde .pt correctamente")

    # PREPROCESSING (igual que tu validación)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])

    val_dir = os.path.join(DATA_ROOT, "validation")
    val_ds = ImageFolder(val_dir, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    correct = 0
    total = 0

    # --- EVALUACIÓN ---
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluando"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"\n✅ Accuracy final del modelo cargado: {acc:.4f}")

if __name__ == "__main__":
    evaluate()
