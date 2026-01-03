import os
from torch.utils.data import DataLoader
from src.fer.data.datasets import AlbumentationsImageFolder, get_transforms


def main():
    base_dir = os.path.join("data", "raw")

    train_dir = os.path.join(base_dir, "train")
    val_dir   = os.path.join(base_dir, "val")
    test_dir  = os.path.join(base_dir, "test")

    train_ds = AlbumentationsImageFolder(train_dir, split="train")
    val_ds   = AlbumentationsImageFolder(val_dir, split="val")
    test_ds  = AlbumentationsImageFolder(test_dir, split="test")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    images, labels = next(iter(train_loader))

    print("Batch imágenes:", images.shape)  # [B, 1, 224, 224]
    print("Batch labels  :", labels.shape)
    print("Ejemplo label:", labels[0].item())


if __name__ == "__main__":
    main()
