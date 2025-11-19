import torch
import torch.nn as nn


class EmotionCNN(nn.Module):
    """
    CNN para clasificar emociones en imágenes 1x224x224.

    Arquitectura:
    - 4 bloques Conv2d + BatchNorm + ReLU + MaxPool
    - AdaptiveAvgPool2d para resumir la parte espacial
    - 2 capas lineales (FC) para clasificar en 7 emociones
    """

    def __init__(self, num_classes: int = 7, in_channels: int = 1):
        super().__init__()

        # Bloque 1: 1x224x224 -> 32x112x112
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Bloque 2: 32x112x112 -> 64x56x56
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Bloque 3: 64x56x56 -> 128x28x28
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Bloque 4: 128x28x28 -> 256x14x14
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Pool global: 256x14x14 -> 256x1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Clasificador: 256 -> 128 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """
        Inicialización Kaiming para que la red empiece en una zona razonable
        del espacio de parámetros y el entrenamiento sea estable y asi nos de
        mejores resultados.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, 1, 224, 224]
        return: logits [batch, num_classes]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.global_pool(x)   # [B, 256, 1, 1]
        x = torch.flatten(x, 1)   # [B, 256]

        x = self.classifier(x)    # [B, num_classes]
        return x




"""Qué está haciendo esta CNN 

Bloques Conv + ReLU + MaxPool
Van detectando patrones cada vez más complejos (bordes → formas → partes de la cara).
Cada MaxPool reduce mitad el tamaño espacial para ahorrar parámetros.

BatchNorm
Normaliza activaciones dentro de cada minibatch → entrenamiento más estable, permite usar LR algo mayor.

AdaptiveAvgPool2d
Resume toda la info espacial a un vector de 256 valores (uno por canal), sin importar el tamaño exacto.

Clasificador FC
Convierte ese vector en probabilidades sobre las 7 emociones."""