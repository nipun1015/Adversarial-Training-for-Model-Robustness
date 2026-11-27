"""Evaluate every trained model on the held-out TEST set under FGSM and PGD.

Reads  models/<name>_seed<seed>.pth   (made by train_experiments.py)
Writes results/all_results.csv         one row per model/seed/attack/epsilon
       results/per_class_accuracy.csv  per-class accuracy at eps = 0.20
       results/confusion_seed<first>.json  confusion matrices at eps = 0.20

Partial runs merge into the existing CSV files (rows of the selected models and
attacks are replaced, everything else is kept):
    python src/evaluate_all.py --models advPGD_eps0.20 --attacks fgsm pgd
    python src/evaluate_all.py --attacks pgd20        (PGD-20 check at eps 0.20, all models)

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

import pandas as pd

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
PGD20_EPS = [0.20]               # extra, stronger check (20 steps) at the main strength
PGD20_STEPS = 20
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
            steps = PGD20_STEPS if attack == "pgd20" else PGD_STEPS
            adv = pgd_attack(model, images, labels, eps, criterion, steps=steps)
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
    parser.add_argument("--models", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--attacks", nargs="+", choices=["fgsm", "pgd", "pgd20"],
                        default=["fgsm", "pgd"])
    args = parser.parse_args()

    torch.manual_seed(0)                       # PGD random start is reproducible
    _, _, test_loader = get_data_loaders()
    criterion = nn.CrossEntropyLoss()
    out = Path("results")
    out.mkdir(exist_ok=True)

    rows, per_class, confusions = [], [], {}
    start = time.time()
    for seed in args.seeds:
        for name in args.models:
            ckpt = Path("models") / f"{name}_seed{seed}.pth"
            if not ckpt.exists():
                print(f"missing {ckpt} - run train_experiments.py first")
                continue
            model = MNISTCNN().to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()

            plan = {"fgsm": FGSM_EPS, "pgd": PGD_EPS, "pgd20": PGD20_EPS}
            for attack in args.attacks:
                eps_list = plan[attack]
                for eps in eps_list:
                    acc, asr, f1, p, y = run_attack(model, test_loader, attack, eps, criterion)
                    rows.append({"model": name, "seed": seed, "attack": attack,
                                 "epsilon": eps, "accuracy": acc,
                                 "attack_success_rate": asr, "macro_f1": f1})
                    if attack != "pgd20" and abs(eps - DETAIL_EPS) < 1e-9:
                        for c in range(10):
                            per_class.append({"model": name, "seed": seed, "attack": attack,
                                              "digit": c,
                                              "accuracy": 100 * ((p == c) & (y == c)).sum() / (y == c).sum()})
                        if seed == args.seeds[0]:
                            confusions[f"{name}|{attack}"] = confusion_matrix(y, p).tolist()
                    print(f"{name} seed {seed} {attack.upper()} eps {eps:.2f}: "
                          f"acc {acc:.2f}% ASR {asr:.2f}% F1 {f1:.2f}%")
    # ---- merge with existing files so partial runs never delete earlier results
    def merge(fname, new_rows, drop):
        path = out / fname
        new_df = pd.DataFrame(new_rows)
        if path.exists():
            old = pd.read_csv(path)
            old = old[~old.apply(drop, axis=1)]
            new_df = pd.concat([old, new_df], ignore_index=True)
        new_df.to_csv(path, index=False)
        return new_df

    done = set(zip([r["model"] for r in rows], [r["seed"] for r in rows], [r["attack"] for r in rows]))
    all_df = merge("all_results.csv", rows, lambda r: (r["model"], r["seed"], r["attack"]) in done)
    done_pc = set(zip([r["model"] for r in per_class], [r["seed"] for r in per_class], [r["attack"] for r in per_class]))
    if per_class:
        merge("per_class_accuracy.csv", per_class, lambda r: (r["model"], r["seed"], r["attack"]) in done_pc)
    cpath = out / f"confusion_seed{args.seeds[0]}.json"
    if cpath.exists():
        confusions = {**json.load(open(cpath)), **confusions}
    with open(cpath, "w") as f:
        json.dump(confusions, f)
    rows = all_df.to_dict("records")
    print(f"\nSaved results/all_results.csv ({len(rows)} rows) in {time.time()-start:.0f}s")


if __name__ == "__main__":
    main()
