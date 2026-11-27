# Adversarial Training for Model Robustness on MNIST

UML501 – Machine Learning Lab (Thapar Institute of Engineering and Technology, Patiala)
Project 6 · Lab Instructor: Charchit Chandra (Research Scholar)

**Team:** Nipun Gupta (1024031169) · Sanchit Narula (1024031048)

## Overview
Deep networks can be fooled by tiny, deliberately crafted perturbations. This project trains a small CNN on MNIST in four ways and tests how well each resists FGSM and PGD attacks:

| Regime | Trained on |
|---|---|
| Baseline | Clean images |
| FGSM-AT ε = 0.10 | FGSM adversarial examples, ε = 0.10 |
| FGSM-AT ε = 0.20 | FGSM adversarial examples, ε = 0.20 |
| FGSM-AT ε = 0.20 + RS | FGSM with a random start (Wong et al., 2020), ε = 0.20 |

Every regime is trained with 3 seeds (42, 43, 44), giving 12 models. All models share one CNN (206,922 parameters), Adam (lr 0.001), batch size 128 and 5 epochs. Data split: 54,000 train / 6,000 validation / 10,000 test (the split always uses seed 42).

## Main findings (mean ± std over 3 seeds, test set)
| ε = 0.20 | Clean acc. | FGSM acc. | PGD-10 acc. |
|---|---|---|---|
| Baseline | 98.69 ± 0.19 | 35.76 ± 3.27 | 1.66 ± 0.32 |
| FGSM-AT 0.10 | 98.77 ± 0.06 | 85.59 ± 5.86 | 22.51 ± 13.17 |
| FGSM-AT 0.20 | 63.36 ± 32.30 | 97.38 ± 3.28 | 1.28 ± 2.21 |
| FGSM-AT 0.20 + RS | 98.39 ± 0.02 | 91.78 ± 0.59 | 76.50 ± 2.69 |

- FGSM adversarial training greatly improves FGSM robustness at almost no cost in clean accuracy.
- A stronger attack (PGD) shows that much of this robustness is not real. FGSM alone overstates it.
- Plain FGSM training at ε = 0.20 is unstable (catastrophic overfitting): in 2 of 3 seeds clean accuracy collapsed. A random start fixes this.
- All models fail against PGD at ε = 0.30.

## Project structure
```
src/
  data.py               MNIST loading, seeded 54k/6k split
  model.py              MNISTCNN architecture
  attacks.py            FGSM, FGSM + random start, PGD
  train_experiments.py  trains 4 regimes x 3 seeds, saves models and logs
  evaluate_all.py       FGSM (7 strengths) and PGD-10 (4 strengths) on the test set
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
Tested with Python 3 and PyTorch (Python 3.14.5 and Name: torchvision Version: 0.29.1). MNIST is downloaded automatically by torchvision on first run. CPU is enough; a CUDA GPU is used if available.

## How to run
Run from the project root:
```bash
python src/test_model.py          # optional smoke tests
python src/test_attack.py
python src/train_experiments.py   # train all 12 models (add --seeds 42 for a quick test, --force to retrain)
python src/evaluate_all.py        # attack all models on the test set
python src/make_plots.py          # figures and summary tables
```
Outputs: `results/all_results.csv`, `results/summary_mean_std.csv`, `results/per_class_accuracy.csv`, `results/training_time_seconds.csv`, `results/logs/*.csv`, `results/plots/*.png`.
Training the 12 models took about 38 minutes (training loops only) on our machine.

## Reproducibility
- Seeds 42, 43, 44 for training; seed 42 for the train/validation split; seed 0 for the PGD random start at evaluation.
- The test set is used only for final evaluation. No checkpoint selection: the last-epoch model is evaluated.
- Re-running seed 42 reproduced our first single-run results exactly.

## Limitations
PGD uses only 10 steps at 3 strengths (so robust accuracies are upper bounds); only white-box attacks; one dataset and one small CNN; only 3 seeds and no significance tests.

## References
1. LeCun et al., "Gradient-based learning applied to document recognition", Proc. IEEE, 1998.
2. Szegedy et al., "Intriguing properties of neural networks", ICLR 2014.
3. Goodfellow et al., "Explaining and harnessing adversarial examples", ICLR 2015.
4. Madry et al., "Towards deep learning models resistant to adversarial attacks", ICLR 2018.
5. Wong, Rice and Kolter, "Fast is better than free: Revisiting adversarial training", ICLR 2020.
6. MNIST database: http://yann.lecun.com/exdb/mnist/

(The full reference list is in the report.)