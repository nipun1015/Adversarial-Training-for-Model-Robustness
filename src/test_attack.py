
import torch
import torch.nn as nn

from src.data import get_data_loaders
from src.model import MNISTCNN
from src.attacks import fgsm_attack


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader = get_data_loaders()

model = MNISTCNN().to(device)
model.load_state_dict(
    torch.load("models/baseline_cnn.pth", map_location=device)
)
model.eval()

criterion = nn.CrossEntropyLoss()
images, labels = next(iter(test_loader))

images, labels = images.to(device), labels.to(device)

for epsilon in [0, 0.05, 0.10, 0.15, 0.20]:
    attacked_images = fgsm_attack(
        model, images, labels, epsilon, criterion
    )

    with torch.no_grad():
        predictions = model(attacked_images).argmax(dim=1)
        accuracy = (predictions == labels).float().mean().item() * 100

    print(
        f"Epsilon: {epsilon:.2f} | "
        f"Accuracy: {accuracy:.2f}% | "
        f"Pixel range: {attacked_images.min().item():.2f} "
        f"to {attacked_images.max().item():.2f}"
    )
