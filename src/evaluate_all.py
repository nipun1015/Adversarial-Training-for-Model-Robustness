"""Evaluate every trained model on the held-out TEST set under FGSM and PGD.

Reads  models/<name>_seed<seed>.pth   (made by train_experiments.py)
Writes results/all_results.csv         one row per model/seed/attack/epsilon
       results/per_class_accuracy.csv  per-class accuracy at eps = 0.20
       results/confusion_seed<first>.json  confusion matrices at eps = 0.20

Metrics: accuracy, attack success rate (ASR, over originally-correct samples),
macro-F1.   Run:  python src/evaluate_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score

from src.attacks import fgsm_attack, pgd_attack
from src.data import get_data_loaders
from src.model import MNISTCNN
from src.train_experiments import CONFIGS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FGSM_EPS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
PGD_EPS = [0.0, 0.10, 0.20, 0.30]   # PGD is ~steps x slower than FGSM, so fewer strengths
PGD_STEPS = 10
DETAIL_EPS = 0.20


def run_attack(model, loader, attack, eps, criterion):
    preds, clean_preds, labels_all = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            clean_preds.append(model(images).argmax(1).cpu())
        if eps == 0:
            adv = images
        elif attack == "fgsm":
            adv = fgsm_attack(model, images, labels, eps, criterion)
        else:
            adv = pgd_attack(model, images, labels, eps, criterion, steps=PGD_STEPS)
        with torch.no_grad():
            preds.append(model(adv).argmax(1).cpu())
        labels_all.append(labels.cpu())
    p, c, y = (torch.cat(t).numpy() for t in (preds, clean_preds, labels_all))
    acc = 100 * (p == y).mean()
    was_correct = c == y
    asr = 100 * ((p != y) & was_correct).sum() / was_correct.sum()
    f1 = 100 * f1_score(y, p, average="macro")
    return acc, asr, f1, p, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    torch.manual_seed(0)                       # PGD random start is reproducible
    _, _, test_loader = get_data_loaders()
    criterion = nn.CrossEntropyLoss()
    out = Path("results")
    out.mkdir(exist_ok=True)

    rows, per_class, confusions = [], [], {}
    start = time.time()
    for seed in args.seeds:
        for name in CONFIGS:
            ckpt = Path("models") / f"{name}_seed{seed}.pth"
            if not ckpt.exists():
                print(f"missing {ckpt} - run train_experiments.py first")
                continue
            model = MNISTCNN().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            for attack, eps_list in (("fgsm", FGSM_EPS), ("pgd", PGD_EPS)):
                for eps in eps_list:
                    acc, asr, f1, p, y = run_attack(model, test_loader, attack, eps, criterion)
                    rows.append({"model": name, "seed": seed, "attack": attack,
                                 "epsilon": eps, "accuracy": acc,
                                 "attack_success_rate": asr, "macro_f1": f1})
                    if abs(eps - DETAIL_EPS) < 1e-9:
                        for c in range(10):
                            per_class.append({"model": name, "seed": seed, "attack": attack,
                                              "digit": c,
                                              "accuracy": 100 * ((p == c) & (y == c)).sum() / (y == c).sum()})
                        if seed == args.seeds[0]:
                            confusions[f"{name}|{attack}"] = confusion_matrix(y, p).tolist()
                    print(f"{name} seed {seed} {attack.upper()} eps {eps:.2f}: "
                          f"acc {acc:.2f}% ASR {asr:.2f}% F1 {f1:.2f}%")
    for fname, data in (("all_results.csv", rows), ("per_class_accuracy.csv", per_class)):
        with open(out / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    with open(out / f"confusion_seed{args.seeds[0]}.json", "w") as f:
        json.dump(confusions, f)
    print(f"\nSaved results/all_results.csv ({len(rows)} rows) in {time.time()-start:.0f}s")


if __name__ == "__main__":
    main()
