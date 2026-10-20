
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_data_loaders
from src.model import MNISTCNN


EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100 * correct / total


def main():
    train_loader, val_loader, test_loader = get_data_loaders()

    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                model.state_dict(),
                model_dir / "baseline_cnn.pth"
            )

    # Load the best checkpoint before final test evaluation
    model.load_state_dict(
        torch.load(model_dir / "baseline_cnn.pth", map_location=device)
    )

    test_loss, test_accuracy = evaluate(model, test_loader, criterion)

    print(f"\nBest validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Clean test accuracy: {test_accuracy:.2f}%")
    print("Checkpoint saved: models/baseline_cnn.pth")


if __name__ == "__main__":
    main()
