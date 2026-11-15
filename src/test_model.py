
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model import MNISTCNN

model = MNISTCNN()

# Create a batch of 128 dummy MNIST images
images = torch.randn(128, 1, 28, 28)

outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))
