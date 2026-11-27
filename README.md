# Adversarial Training for Model Robustness on MNIST

**Team:** Nipun Gupta (1024031169) · Sanchit Narula (1024031048)

## Overview
Deep networks can be fooled by tiny, deliberately crafted perturbations. This project trains a small CNN on MNIST in five ways and tests how well each resists FGSM and PGD attacks:

| Regime | Trained on |
|---|---|
| Baseline | Clean images |
| FGSM-AT ε = 0.10 | FGSM adversarial examples, ε = 0.10 |
| FGSM-AT ε = 0.20 | FGSM adversarial examples, ε = 0.20 |
| FGSM-AT ε = 0.20 + RS | FGSM with a random start (Wong et al., 2020), ε = 0.20 |
| PGD-AT ε = 0.20 | PGD-10 adversarial examples (Madry et al., 2018), ε = 0.20 |

Every regime is trained with 3 seeds (42, 43, 44), giving 15 models. All models share one CNN (206,922 parameters), Adam (lr 0.001), batch size 128 and 5 epochs. Data split: 54,000 train / 6,000 validation / 10,000 test (the split always uses seed 42).

## Main findings (mean ± std over 3 seeds, test set, ε = 0.20)
| Regime | Clean acc. | FGSM acc. | PGD-10 acc. | PGD-20 acc. | Train time (s) |
|---|---|---|---|---|---|
| Baseline | 98.69 ± 0.19 | 35.76 ± 3.27 | 1.66 ± 0.32 | 1.21 ± 0.30 | 127.2 |
| FGSM-AT 0.10 | 98.77 ± 0.06 | 85.59 ± 5.86 | 22.51 ± 13.17 | 19.89 ± 12.09 | 211.2 |
| FGSM-AT 0.20 | 63.36 ± 32.30 | 97.38 ± 3.28 | 1.28 ± 2.21 | 0.99 ± 1.71 | 220.3 |
| FGSM-AT 0.20 + RS | 98.39 ± 0.02 | 91.78 ± 0.59 | 76.50 ± 2.69 | 74.95 ± 3.35 | 206.4 |
| PGD-AT 0.20 | 97.76 ± 0.12 | 91.39 ± 0.87 | 88.91 ± 0.92 | 88.66 ± 0.87 | 634.7 |

- FGSM adversarial training greatly improves FGSM robustness, but a stronger attack (PGD) shows that much of this robustness is not real. FGSM-only evaluation overstates it.
- Plain FGSM training at ε = 0.20 is unstable (catastrophic overfitting): clean accuracy collapsed in 2 of 3 seeds. A random start fixes this.
- PGD-10 adversarial training is the most robust (88.91% under PGD-10) at a cost of about 1 point of clean accuracy and about 5× the training time of the baseline.
- PGD-20 lowers PGD-AT accuracy by only 0.25 points, so there is no sign of gradient masking.
- Robustness stays tied to the training budget: PGD-AT falls to 9.67% at ε = 0.30.

## Project structure
```
src/
  data.py               MNIST loading, seeded 54k/6k split
  model.py              MNISTCNN architecture
  attacks.py            FGSM, FGSM + random start, PGD
  train_experiments.py  trains 5 regimes x 3 seeds (--only <regime> trains one), saves models and logs
  evaluate_all.py       FGSM (7 strengths), PGD-10 (4 strengths), PGD-20 check; merges into existing CSVs
  make_plots.py         figures and mean ± std summary
  train.py, adversarial_train.py, evaluate.py, visualize_attacks.py   original single-run scripts
  test_model.py, test_attack.py                                       smoke tests
notebooks/01_mnist_eda.ipynb   exploratory data analysis
models/                        saved checkpoints (<regime>_seed<seed>.pth)
results/                       CSV results, logs/ and plots/
report/                        project report
```

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.14.5 and torchvision 0.29.1. MNIST is downloaded automatically by torchvision on first run. CPU is enough; a CUDA GPU is used if available.

## How to run
Run from the project root:
```bash
python src/test_model.py          # optional smoke tests
python src/test_attack.py
python src/train_experiments.py   # train all 15 models (--seeds 42 for a quick test, --force to retrain)
python src/evaluate_all.py        # FGSM and PGD-10 on the test set
python src/evaluate_all.py --attacks pgd20   # PGD-20 check at eps = 0.20
python src/make_plots.py          # figures and summary tables
```
To add or retrain a single regime: `python src/train_experiments.py --only advPGD_eps0.20`, then `python src/evaluate_all.py --models advPGD_eps0.20 --attacks fgsm pgd`. Partial runs merge into the existing CSV files.

Outputs: `results/all_results.csv`, `results/summary_mean_std.csv`, `results/per_class_accuracy.csv`, `results/training_time_seconds.csv`, `results/logs/*.csv`, `results/plots/*.png`.
Training the 15 models took about 70 minutes (training loops only) on our machine.

## Reproducibility
- Seeds 42, 43, 44 for training; seed 42 for the train/validation split; seed 0 for the PGD random start at evaluation.
- The test set is used only for final evaluation. No checkpoint selection: the last-epoch model is evaluated.
- Re-running seed 42 reproduced our first single-run results exactly.

## Limitations
PGD uses only 10 (and 20) steps with no random restarts, so robust accuracies are upper bounds; only white-box L∞ attacks; one dataset and one small CNN; only 3 seeds and no significance tests.

## References
1. LeCun et al., "Gradient-based learning applied to document recognition", Proc. IEEE, 1998.
2. Szegedy et al., "Intriguing properties of neural networks", ICLR 2014.
3. Goodfellow et al., "Explaining and harnessing adversarial examples", ICLR 2015.
4. Madry et al., "Towards deep learning models resistant to adversarial attacks", ICLR 2018.
5. Wong, Rice and Kolter, "Fast is better than free: Revisiting adversarial training", ICLR 2020.
6. Andriushchenko and Flammarion, "Understanding and improving fast adversarial training", NeurIPS 2020.
7. Athalye, Carlini and Wagner, "Obfuscated gradients give a false sense of security", ICML 2018.

The full report is in `report/`.