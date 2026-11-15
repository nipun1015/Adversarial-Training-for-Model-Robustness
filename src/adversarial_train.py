import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_data_loaders
from src.model import MNISTCNN
from src.attacks import fgsm_attack


SEED = 42
EPOCHS = 5
LEARNING_RATE = 0.001
TRAIN_EPSILON = 0.10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_clean(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def main():
    torch.manual_seed(SEED)

    train_loader, val_loader, _ = get_data_loaders()

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = 0.0
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            adversarial_images = fgsm_attack(
                model,
                images,
                labels,
                TRAIN_EPSILON,
                criterion
            )

            optimizer.zero_grad()
            outputs = model(adversarial_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_accuracy = evaluate_clean(model, val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Clean Validation Accuracy: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                model.state_dict(),
                model_dir / "adversarial_cnn.pth"
            )

    print(f"\nBest clean validation accuracy: {best_val_accuracy:.2f}%")
    print("Checkpoint saved: models/adversarial_cnn.pth")


if __name__ == "__main__":
    main()
