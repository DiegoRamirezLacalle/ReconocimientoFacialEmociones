import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from transformers import ViTForImageClassification
from tqdm import tqdm

# --- CONFIGURACIÓN --- --
MODEL_NAME = "mo-thecreator/vit-Facial-Expression-Recognition"
NUM_CLASSES = 7
DATA_ROOT = "data" 
BATCH_SIZE = 16    
EPOCHS = 10        
LR = 1e-5          # <--- CAMBIO: Bajamos el LR (antes 2e-5) para que sea más estable
MODEL_OUT = r"C:\Users\joorg\Documents\ReconocimientoFacialEmociones\notebooks\vit_emotions.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

def validate_model(model, loader, desc="Validando"):
    """ Función auxiliar para evaluar el modelo y devolver el accuracy """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).logits
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

def main():
    print(f"Usando dispositivo: {device}")

    # 1. CARGAR ARQUITECTURA BASE
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    ).to(device)

    # 2. PREPARAR DATOS
    transform_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
    ])

    transform_val = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    train_ds = ImageFolder(os.path.join(DATA_ROOT, "train"), transform=transform_train)
    val_ds = ImageFolder(os.path.join(DATA_ROOT, "validation"), transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 3. SISTEMA DE SEGURIDAD (RÉCORD A BATIR)
    best_acc = 0.0

    if os.path.exists(MODEL_OUT):
        print(f"\n⚠️ Encontrado modelo existente: {MODEL_OUT}")
        print("Comprobando su accuracy actual para no perder calidad...")
        try:
            # Cargamos el estado actual del archivo
            model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
            best_acc = validate_model(model, val_loader, desc="Checkeo Inicial")
            print(f"✅ Récord actual a batir: {best_acc:.4f}")
        except Exception as e:
            print(f"❌ Error leyendo el modelo anterior, empezamos de 0. Error: {e}")
            best_acc = 0.0
    else:
        print("\nℹ️ No existe modelo previo, el récord a batir es 0.0")

    # 4. CONFIGURAR ENTRENAMIENTO
    # Importante: Como cargamos pesos para el checkeo, nos aseguramos de que el modelo
    # esté listo para entrenar (aunque partimos de los pesos pre-entrenados del repo original
    # o del .pt si decidimos continuar desde ahí. 
    # NOTA: Para reentrenar "limpio" pero superando el récord, volvemos a cargar el base de HuggingFace
    # Si quisieras continuar el entrenamiento del .pt, comenta las siguientes líneas.
    
    print("\n🔄 Reiniciando pesos desde base HuggingFace para reentrenamiento limpio...")
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_CLASSES, ignore_mismatched_sizes=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

    print(f"\n🚀 Iniciando entrenamiento ({EPOCHS} épocas). Solo guardaremos si superamos: {best_acc:.4f}")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # --- TRAINING ---
        model.train()
        total_loss = 0
        
        for images, labels in tqdm(train_loader, desc="Train"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # --- VALIDATION ---
        acc = validate_model(model, val_loader, desc="Val")
        print(f"Loss: {avg_loss:.4f} | Val acc: {acc:.4f} (Récord: {best_acc:.4f})")

        # --- GUARDADO CONDICIONAL ---
        if acc > best_acc:
            print(f"🔥 ¡NUEVO RÉCORD! ({acc:.4f} > {best_acc:.4f}). Actualizando {MODEL_OUT}...")
            best_acc = acc
            torch.save(model.state_dict(), MODEL_OUT)
        else:
            print(f"📉 No mejora el récord ({acc:.4f} <= {best_acc:.4f}). No guardamos.")

if __name__ == "__main__":
    main()