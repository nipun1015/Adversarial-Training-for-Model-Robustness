"""Train the baseline and adversarially trained CNNs for several seeds and
log everything per epoch.

Models trained (same architecture, optimiser, epochs, batch size, data split):
    baseline      : clean images
    adv_eps0.10   : FGSM adversarial training, eps_train = 0.10
    adv_eps0.20   : FGSM adversarial training, eps_train = 0.20
    advRS_eps0.20 : FGSM with random start (Wong et al. 2020), eps_train = 0.20
    advPGD_eps0.20: PGD-10 adversarial training (Madry et al. 2018), eps_train = 0.20

Outputs
    models/<name>_seed<seed>.pth            checkpoint after the LAST epoch
    results/logs/<name>_seed<seed>.csv      per-epoch log (loss, accuracies, time)

No checkpoint selection is used: every model is simply the result of 5 epochs.
(This avoids hiding unstable training, and never touches the test set.)
Run:  python src/train_experiments.py            (add --force to retrain)
      python src/train_experiments.py --only advPGD_eps0.20   (train just one regime)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.attacks import fgsm_attack, fgsm_rs_attack, pgd_attack
from src.data import get_data_loaders
from src.model import MNISTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 0.001
CONFIGS = {              # name -> (eps_train, training attack)
    "baseline": (0.0, None),
    "adv_eps0.10": (0.10, "fgsm"),
    "adv_eps0.20": (0.20, "fgsm"),
    "advRS_eps0.20": (0.20, "fgsm_rs"),
    "advPGD_eps0.20": (0.20, "pgd"),
}
TRAIN_PGD_STEPS = 10      # steps of the PGD attack used *during training*
MONITOR_EPS = (0.10, 0.20)   # FGSM strengths logged on the validation set for every model

MODEL_DIR = Path("models")
LOG_DIR = Path("results") / "logs"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def val_accuracy(model, loader, criterion, eps):
    """Accuracy on a loader, optionally under FGSM with strength eps."""
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if eps > 0:
            images = fgsm_attack(model, images, labels, eps, criterion)
        with torch.no_grad():
            correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100 * correct / total


def train_one(name, eps_train, method, seed, epochs):
    set_seed(seed)
    train_loader, val_loader, _ = get_data_loaders(train_seed=seed)
    model = MNISTCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ckpt = MODEL_DIR / f"{name}_seed{seed}.pth"
    log_path = LOG_DIR / f"{name}_seed{seed}.csv"
    rows, run_start = [], time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if method == "fgsm":                    # adversarial training
                images = fgsm_attack(model, images, labels, eps_train, criterion)
            elif method == "fgsm_rs":
                images = fgsm_rs_attack(model, images, labels, eps_train, criterion)
            elif method == "pgd":
                images = pgd_attack(model, images, labels, eps_train, criterion,
                                    steps=TRAIN_PGD_STEPS)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        train_secs = time.time() - t0
        v_clean = val_accuracy(model, val_loader, criterion, 0.0)
        v_mon = {e: val_accuracy(model, val_loader, criterion, e) for e in MONITOR_EPS}

        row = {"model": name, "seed": seed, "epoch": epoch,
               "train_loss": loss_sum / seen,
               "train_acc_on_training_inputs": 100 * correct / seen,
               "val_clean_acc": v_clean}
        for e in MONITOR_EPS:
            row[f"val_fgsm_eps{e:.2f}_acc"] = v_mon[e]
        row["epoch_train_seconds"] = train_secs
        rows.append(row)
        print(f"[{name} | seed {seed}] epoch {epoch}/{epochs} loss {loss_sum/seen:.4f} | "
              f"val clean {v_clean:.2f}% | val FGSM@0.10 {v_mon[0.10]:.2f}% | "
              f"val FGSM@0.20 {v_mon[0.20]:.2f}% | {train_secs:.1f}s")

    torch.save(model.state_dict(), ckpt)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"   saved {ckpt} and {log_path} (total {time.time()-run_start:.0f}s)\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--only", nargs="+", choices=list(CONFIGS), default=None,
                        help="train only these regimes (default: all)")
    parser.add_argument("--force", action="store_true", help="retrain even if checkpoint exists")
    args = parser.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | torch {torch.__version__}")

    for seed in args.seeds:
        for name, (eps_train, method) in CONFIGS.items():
            if args.only and name not in args.only:
                continue
            ckpt = MODEL_DIR / f"{name}_seed{seed}.pth"
            if ckpt.exists() and not args.force:
                print(f"skip {ckpt} (exists; use --force to retrain)")
                continue
            train_one(name, eps_train, method, seed, args.epochs)


if __name__ == "__main__":
    main()
