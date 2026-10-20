
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
