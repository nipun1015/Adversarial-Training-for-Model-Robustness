"""Create all report figures and the mean +/- std summary table.

Reads  results/all_results.csv, results/per_class_accuracy.csv, results/logs/*.csv
Writes results/plots/*.png  and  results/summary_mean_std.csv

Run:  python src/make_plots.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path("results")
PLOTS = RES / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

ORDER = ["baseline", "adv_eps0.10", "adv_eps0.20", "advRS_eps0.20"]
LABEL = {"baseline": "Baseline (clean training)",
         "adv_eps0.10": "Adversarial training, ε = 0.10",
         "adv_eps0.20": "Adversarial training, ε = 0.20",
         "advRS_eps0.20": "Adv. training ε = 0.20 + random start"}
COLOR = {"baseline": "#b04a3f", "adv_eps0.10": "#2f6f8f", "adv_eps0.20": "#3a8a5a", "advRS_eps0.20": "#c98a1a"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 120})

df = pd.read_csv(RES / "all_results.csv")
n_seeds = df["seed"].nunique()

# ---------------------------------------------------------------- summary table
g = df.groupby(["model", "attack", "epsilon"])[["accuracy", "attack_success_rate", "macro_f1"]]
summary = g.agg(["mean", "std"]).round(2)
summary.columns = [f"{a}_{b}" for a, b in summary.columns]
summary = summary.reset_index()
summary.to_csv(RES / "summary_mean_std.csv", index=False)
print(f"Seeds found: {sorted(df['seed'].unique())}")


def curve(metric, attack, fname, ylabel, title, ylim=(0, 100)):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for m in ORDER:
        s = df[(df.model == m) & (df.attack == attack)].groupby("epsilon")[metric].agg(["mean", "std"]).fillna(0)
        ax.plot(s.index, s["mean"], marker="o", ms=4, color=COLOR[m], label=LABEL[m])
        ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=COLOR[m], alpha=0.18)
    ax.set_xlabel("Perturbation strength ε")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (mean ± std over {n_seeds} seeds)")
    ax.set_ylim(*ylim)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(PLOTS / fname, dpi=200)
    plt.close(fig)


curve("accuracy", "fgsm", "fgsm_accuracy_vs_eps.png", "Test accuracy (%)", "FGSM attack")
curve("accuracy", "pgd", "pgd_accuracy_vs_eps.png", "Test accuracy (%)", "PGD-10 attack")
curve("attack_success_rate", "fgsm", "fgsm_asr_vs_eps.png", "Attack success rate (%)", "FGSM attack")
curve("attack_success_rate", "pgd", "pgd_asr_vs_eps.png", "Attack success rate (%)", "PGD-10 attack")

# ---------------------------------------------------------------- FGSM vs PGD per model
fig, axes = plt.subplots(1, 4, figsize=(14, 3.7), sharey=True)
for ax, m in zip(axes, ORDER):
    for atk, ls in (("fgsm", "-"), ("pgd", "--")):
        s = df[(df.model == m) & (df.attack == atk)].groupby("epsilon")["accuracy"].mean()
        ax.plot(s.index, s.values, ls, marker="o", ms=3.5, color=COLOR[m], label=atk.upper() if atk == "fgsm" else "PGD-10")
    ax.set_title(LABEL[m], fontsize=9)
    ax.set_xlabel("ε")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("Test accuracy (%)")
fig.tight_layout()
fig.savefig(PLOTS / "fgsm_vs_pgd_per_model.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- trade-off scatter
fig, ax = plt.subplots(figsize=(9.2, 4.6))
for m in ORDER:
    clean = df[(df.model == m) & (df.attack == "fgsm") & (df.epsilon == 0)].accuracy
    for atk, mk in (("fgsm", "o"), ("pgd", "s")):
        rob = df[(df.model == m) & (df.attack == atk) & (df.epsilon == 0.20)].accuracy
        ax.errorbar(clean.mean(), rob.mean(), xerr=clean.std() if len(clean) > 1 else 0,
                    yerr=rob.std() if len(rob) > 1 else 0, fmt=mk, color=COLOR[m], ms=8, capsize=3,
                    label=f"{LABEL[m]} – {'FGSM' if atk == 'fgsm' else 'PGD-10'}")
ax.set_xlabel("Clean test accuracy (%)")
ax.set_ylabel("Accuracy under attack, ε = 0.20 (%)")
ax.set_title("Clean accuracy vs robustness trade-off")
ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout()
fig.savefig(PLOTS / "tradeoff_clean_vs_robust.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- per-class bars
pc = pd.read_csv(RES / "per_class_accuracy.csv")
for atk in ("fgsm", "pgd"):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.2
    for k, m in enumerate(ORDER):
        s = pc[(pc.model == m) & (pc.attack == atk)].groupby("digit")["accuracy"].mean()
        ax.bar(s.index + (k - 1.5) * w, s.values, w, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(range(10))
    ax.set_xlabel("True digit class")
    ax.set_ylabel("Accuracy at ε = 0.20 (%)")
    ax.set_title(f"Per-class accuracy under {'FGSM' if atk == 'fgsm' else 'PGD-10'}, ε = 0.20 (mean of {n_seeds} seeds)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    fig.savefig(PLOTS / f"per_class_{atk}_eps020.png", dpi=200)
    plt.close(fig)

# ---------------------------------------------------------------- learning curves
logs = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(RES / "logs" / "*.csv")))])
mon = [c for c in logs.columns if c.startswith("val_fgsm")][0]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
mon20 = "val_fgsm_eps0.20_acc"
panels = [("train_loss", "Training loss"), ("val_clean_acc", "Validation accuracy, clean (%)"),
          (mon, "Validation accuracy under FGSM ε = 0.10 (%)"), (mon20, "Validation accuracy under FGSM ε = 0.20 (%)")]
for ax, (col, title) in zip(axes, panels):
    for m_ in ORDER:
        d_ = logs[logs.model == m_]
        for _, ds in d_.groupby("seed"):
            ax.plot(ds.epoch, ds[col], color=COLOR[m_], alpha=0.30, lw=1)
        ax.plot(sorted(d_.epoch.unique()), d_.groupby("epoch")[col].mean(), marker="o", ms=4,
                color=COLOR[m_], label=LABEL[m_], lw=2)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("Epoch")
    ax.set_xticks(sorted(logs.epoch.unique()))
axes[0].legend(frameon=False, fontsize=7.5)
fig.tight_layout()
fig.savefig(PLOTS / "learning_curves.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- training cost
cost = logs.groupby(["model", "seed"])["epoch_train_seconds"].sum().groupby("model").agg(["mean", "std"]).round(1)
cost.to_csv(RES / "training_time_seconds.csv")
fig, ax = plt.subplots(figsize=(5.8, 3.6))
cost = cost.reindex(ORDER)
ax.bar(range(4), cost["mean"], yerr=cost["std"].fillna(0), color=[COLOR[m] for m in ORDER], capsize=3)
ax.set_xticks(range(4))
ax.set_xticklabels(["Baseline", "Adv. 0.10", "Adv. 0.20", "Adv. 0.20+RS"], fontsize=8)
ax.set_ylabel("Training time, 5 epochs (s)")
ax.set_title("Cost of adversarial training")
fig.tight_layout()
fig.savefig(PLOTS / "training_time.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- console table
print("\nTest accuracy (%), mean ± std")
for atk in ("fgsm", "pgd"):
    print(f"\n{atk.upper()}")
    t = summary[summary.attack == atk].copy()
    t["acc"] = t.accuracy_mean.map("{:.2f}".format) + " ± " + t.accuracy_std.fillna(0).map("{:.2f}".format)
    print(t.pivot(index="epsilon", columns="model", values="acc")[ORDER].to_string())
print("\nSaved plots to results/plots/ and results/summary_mean_std.csv")
