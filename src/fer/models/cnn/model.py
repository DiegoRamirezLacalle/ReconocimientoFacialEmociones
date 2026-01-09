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

class ResidualBlock(nn.Module):
    """
    Bloque residual básico tipo ResNet:
    Conv-BN-ReLU -> Conv-BN + conexión de identidad (skip connection).
    Si cambia el nº de canales o el stride, se usa un atajo (downsample).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Si cambia tamaño espacial (stride != 1) o canales, necesitamos adaptar la rama de identidad
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity   # conexión residual
        out = self.relu(out)
        return out


class EmotionResNetSmall(nn.Module):
    """
    ResNet pequeña para FER-2013 (1x224x224):

    - Stem inicial: Conv7x7 + BN + ReLU + MaxPool
    - 4 "layers" residuales:
        * layer1: 64 canales, 2 bloques
        * layer2: 128 canales, 2 bloques (downsample)
        * layer3: 256 canales, 2 bloques (downsample)
        * layer4: 512 canales, 2 bloques (downsample)
    - Global Avg Pool + FC -> 7 emociones
    """

    def __init__(self, num_classes: int = 7, in_channels: int = 1):
        super().__init__()

        # "Stem" inicial (como en ResNet)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Ahora el tamaño es aprox 64 x 56 x 56

        # Layers residuales
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)   # 64x56x56
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)  # 128x28x28
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2) # 256x14x14
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2) # 512x7x7

        # Pool global + clasificador
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 512x1x1
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """
        Crea una "layer" residual con:
        - 1er bloque con posible downsample (stride != 1 o cambio de canales)
        - resto de bloques con stride=1
        """
        blocks = []
        # Primer bloque: puede cambiar tamaño y canales
        blocks.append(ResidualBlock(in_channels, out_channels, stride=stride))
        # Resto de bloques: mismo nº de canales, stride=1
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        """
        Inicialización Kaiming adecuada para ReLU.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, 224, 224]
        x = self.stem(x)     # [B, 64, 56, 56]
        x = self.layer1(x)   # [B, 64, 56, 56]
        x = self.layer2(x)   # [B, 128, 28, 28]
        x = self.layer3(x)   # [B, 256, 14, 14]
        x = self.layer4(x)   # [B, 512, 7, 7]

        x = self.avgpool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        x = self.fc(x)       # [B, num_classes]
        return x
