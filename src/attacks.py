import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch


def fgsm_attack(model, images, labels, epsilon, criterion):
    images = images.detach().clone().requires_grad_(True)

    outputs = model(images)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    perturbation = epsilon * images.grad.sign()
    attacked_images = images + perturbation

    return torch.clamp(attacked_images, 0, 1).detach()
