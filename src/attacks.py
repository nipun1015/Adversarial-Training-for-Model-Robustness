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


def pgd_attack(model, images, labels, epsilon, criterion,
               steps=20, alpha=None, random_start=True):
    """Projected Gradient Descent (Madry et al.): a multi-step, stronger
    version of FGSM. Each step moves alpha in the gradient-sign direction and
    projects back into the L-infinity ball of radius epsilon around the
    original image (and into the valid pixel range [0, 1])."""
    if alpha is None:
        alpha = 2.5 * epsilon / steps

    original = images.detach().clone()
    adv = original.clone()

    if random_start:
        adv = adv + torch.empty_like(adv).uniform_(-epsilon, epsilon)
        adv = torch.clamp(adv, 0, 1)

    for _ in range(steps):
        adv.requires_grad_(True)
        loss = criterion(model(adv), labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.min(torch.max(adv, original - epsilon), original + epsilon)
        adv = torch.clamp(adv, 0, 1)

    return adv.detach()


def fgsm_rs_attack(model, images, labels, epsilon, criterion, alpha=None):
    """FGSM with a random start ("fast adversarial training", Wong et al. 2020).
    The image is first moved to a random point inside the epsilon-ball, then one
    larger gradient-sign step (alpha = 1.25 * epsilon) is taken and the result is
    projected back into the epsilon-ball. The random start stops the model from
    over-fitting to the exact FGSM direction (catastrophic overfitting)."""
    if alpha is None:
        alpha = 1.25 * epsilon

    original = images.detach().clone()
    start = original + torch.empty_like(original).uniform_(-epsilon, epsilon)
    start = torch.clamp(start, 0, 1).requires_grad_(True)

    loss = criterion(model(start), labels)
    grad = torch.autograd.grad(loss, start)[0]

    adv = start.detach() + alpha * grad.sign()
    adv = torch.min(torch.max(adv, original - epsilon), original + epsilon)
    return torch.clamp(adv, 0, 1).detach()
