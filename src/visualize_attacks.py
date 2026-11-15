import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathlib import Path

import torch
import matplotlib.pyplot as plt

from src.data import get_data_loaders
from src.model import MNISTCNN
from src.attacks import fgsm_attack


EPSILON = 0.20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint):
    model = MNISTCNN().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def main():
    _, _, test_loader = get_data_loaders()

    baseline = load_model("models/baseline_cnn.pth")
    robust = load_model("models/adversarial_cnn.pth")

    criterion = torch.nn.CrossEntropyLoss()
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    clean_images = images.detach()

    with torch.no_grad():
        clean_predictions = baseline(clean_images).argmax(dim=1)

    clean_correct = clean_predictions == labels
    images = images[clean_correct][:32]
    labels = labels[clean_correct][:32]

    baseline_adv = fgsm_attack(
        baseline, images, labels, EPSILON, criterion
    )
    robust_adv = fgsm_attack(
        robust, images, labels, EPSILON, criterion
    )

    with torch.no_grad():
        baseline_predictions = baseline(baseline_adv).argmax(dim=1)
        robust_predictions = robust(robust_adv).argmax(dim=1)

    both_defended = []
    baseline_fooled_robust_defended = []
    both_fooled = []

    for i in range(len(labels)):
        b_wrong = baseline_predictions[i] != labels[i]
        r_wrong = robust_predictions[i] != labels[i]

        if b_wrong and not r_wrong:
            baseline_fooled_robust_defended.append(i)
        elif not b_wrong and not r_wrong:
            both_defended.append(i)
        elif b_wrong and r_wrong:
            both_fooled.append(i)

    # Showcase successful defense, unperturbed/resisted, and mutual vulnerability
    selected = (
        baseline_fooled_robust_defended[:5] +
        both_defended[:2] +
        both_fooled[:1]
    )

    if len(selected) < 8:
        for i in range(len(labels)):
            if i not in selected:
                selected.append(i)
            if len(selected) == 8:
                break

    fig, axes = plt.subplots(len(selected), 3, figsize=(11, 2.7 * len(selected)))

    if len(selected) == 1:
        axes = [axes]

    for row, index in enumerate(selected):
        true_label = labels[index].item()
        base_pred = baseline_predictions[index].item()
        rob_pred = robust_predictions[index].item()

        base_status = "Attack Succeeded" if base_pred != true_label else "Attack Failed"
        rob_status = "Attack Defended" if rob_pred == true_label else "Attack Succeeded"

        panels = [
            (images[index], f"Original Clean\nTrue: {true_label}"),
            (
                baseline_adv[index],
                f"Baseline FGSM (eps={EPSILON})\nPred: {base_pred} ({base_status})"
            ),
            (
                robust_adv[index],
                f"Robust FGSM (eps={EPSILON})\nPred: {rob_pred} ({rob_status})"
            )
        ]

        for col, (image, title) in enumerate(panels):
            axes[row][col].imshow(
                image.detach().cpu().squeeze(),
                cmap="gray",
                vmin=0,
                vmax=1
            )
            color = "darkred" if "Succeeded" in title else ("darkgreen" if ("Defended" in title or "Failed" in title) else "black")
            axes[row][col].set_title(title, fontsize=10, color=color if col > 0 else "black")
            axes[row][col].axis("off")

    fig.suptitle(f"FGSM Adversarial Attacks: Baseline vs Robust CNN (epsilon={EPSILON})", fontsize=13, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "adversarial_examples.png"

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"Number of examples displayed: {len(selected)}")


if __name__ == "__main__":
    main()
