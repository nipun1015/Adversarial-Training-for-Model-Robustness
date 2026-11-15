import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

MODELS = {
    "Baseline": "models/baseline_cnn.pth",
    "Adversarially Trained": "models/adversarial_cnn.pth"
}


def evaluate_model(model, test_loader, epsilon, criterion):
    correct = 0
    total = 0
    originally_correct = 0
    successful_attacks = 0

    model.eval()

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_predictions = model(images).argmax(dim=1)

        clean_correct = clean_predictions == labels
        originally_correct += clean_correct.sum().item()

        if epsilon == 0:
            attacked_images = images
        else:
            attacked_images = fgsm_attack(
                model, images, labels, epsilon, criterion
            )

        with torch.no_grad():
            predictions = model(attacked_images).argmax(dim=1)

        correct += (predictions == labels).sum().item()
        successful_attacks += (
            clean_correct & (predictions != labels)
        ).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    attack_success_rate = (
        100 * successful_attacks / originally_correct
        if originally_correct > 0 else 0.0
    )

    return accuracy, attack_success_rate


def main():
    _, _, test_loader = get_data_loaders()
    criterion = nn.CrossEntropyLoss()
    results = []

    for model_name, checkpoint in MODELS.items():
        model = MNISTCNN().to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device)
        )

        print(f"\nEvaluating {model_name} model")

        for epsilon in EPSILONS:
            accuracy, attack_success_rate = evaluate_model(
                model, test_loader, epsilon, criterion
            )

            results.append({
                "model": model_name,
                "epsilon": epsilon,
                "accuracy": accuracy,
                "attack_success_rate": attack_success_rate
            })

            print(
                f"Epsilon: {epsilon:.2f} | "
                f"Accuracy: {accuracy:.2f}% | "
                f"Attack Success Rate: {attack_success_rate:.2f}%"
            )

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "model_comparison.csv"

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "epsilon",
                "accuracy",
                "attack_success_rate"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    for metric, ylabel, title, filename in [
        (
            "accuracy",
            "Test Accuracy (%)",
            "Baseline vs Adversarial Training: FGSM Accuracy",
            "model_accuracy_comparison.png"
        ),
        (
            "attack_success_rate",
            "Attack Success Rate (%)",
            "Baseline vs Adversarial Training: Attack Success Rate",
            "attack_success_comparison.png"
        )
    ]:
        plt.figure()

        for model_name in MODELS:
            model_rows = [
                row for row in results
                if row["model"] == model_name
            ]

            x_values = [row["epsilon"] for row in model_rows]
            y_values = [row[metric] for row in model_rows]

            plt.plot(
                x_values,
                y_values,
                marker="o",
                label=model_name
            )

        plt.xlabel("Epsilon")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.ylim(0, 100)
        plt.xticks(EPSILONS)
        plt.grid(True)
        plt.legend()
        plt.savefig(
            output_dir / filename,
            dpi=200,
            bbox_inches="tight"
        )
        plt.close()

        print(f"Saved: results/{filename}")

    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
