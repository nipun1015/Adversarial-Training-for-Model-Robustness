
import csv
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.data import get_data_loaders
from src.model import MNISTCNN
from src.attacks import fgsm_attack


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILONS = [0, 0.05, 0.10, 0.15, 0.20]


def main():
    _, _, test_loader = get_data_loaders()

    model = MNISTCNN().to(device)
    model.load_state_dict(
        torch.load("models/baseline_cnn.pth", map_location=device)
    )
    model.eval()

    criterion = nn.CrossEntropyLoss()
    results = []

    for epsilon in EPSILONS:
        correct = 0
        total = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            if epsilon == 0:
                attacked_images = images
            else:
                attacked_images = fgsm_attack(
                    model, images, labels, epsilon, criterion
                )

            with torch.no_grad():
                predictions = model(attacked_images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        results.append((epsilon, accuracy))

        print(f"Epsilon: {epsilon:.2f} | Accuracy: {accuracy:.2f}%")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "fgsm_accuracy.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epsilon", "accuracy"])
        writer.writerows(results)

    epsilons = [row[0] for row in results]
    accuracies = [row[1] for row in results]

    plt.plot(epsilons, accuracies, marker="o")
    plt.xlabel("Epsilon")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Baseline CNN Accuracy under FGSM Attack")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(output_dir / "fgsm_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Saved: results/fgsm_accuracy.csv")
    print("Saved: results/fgsm_accuracy.png")


if __name__ == "__main__":
    main()
