# Adversarial Training for Model Robustness on MNIST

**Course:** UML501 – Machine Learning Lab  
**Institution:** Thapar Institute of Engineering and Technology, Patiala  
**Lab Instructor:** Charchit Chandra (Research Scholar)  
**Project Selection:** Project 6: Adversarial Training for Model Robustness  

---

## 1. Project Overview

Deep neural networks achieve high accuracy on standard computer vision benchmarks but exhibit extreme vulnerability to small, imperceptible, adversarially engineered perturbations. This project investigates the vulnerability of a Convolutional Neural Network (CNN) on the MNIST handwritten-digit benchmark under the **Fast Gradient Sign Method (FGSM)** attack, and quantitatively evaluates **adversarial training** as an empirical defense mechanism to improve model robustness.

Two models are developed, trained, and compared under identical architectures and optimization conditions:
- **Baseline CNN:** Trained strictly on clean MNIST training images.
- **Adversarially Trained CNN:** Trained dynamically using FGSM adversarial examples generated on-the-fly at perturbation strength $\epsilon = 0.10$.

Both models are comprehensively evaluated across a spectrum of perturbation strengths ($\epsilon \in \{0.00, 0.05, 0.10, 0.15, 0.20\}$) measuring clean test accuracy, perturbed test accuracy, and attack success rate (ASR).

---

## 2. Research Questions and Objectives

The key research questions and objectives addressed in this project are:
1. **Vulnerability Quantification:** How rapidly does a standard baseline CNN degrade in classification accuracy as the FGSM perturbation strength $\epsilon$ increases from $0.00$ to $0.20$?
2. **Adversarial Training Defense:** Does augmenting the training pipeline with FGSM-perturbed images ($\epsilon = 0.10$) confer significant resistance against white-box gradient-based attacks across multiple unseen attack strengths?
3. **Robustness vs. Clean Accuracy Trade-off:** Does conferring adversarial robustness induce a penalty on clean test data accuracy?
4. **Failure Case Inspection:** How do the models fail under strong perturbations, and how do adversarial perturbations push digit images across decision boundaries into geometrically adjacent classes?

---

## 3. Dataset Documentation & Preprocessing Pipeline

### 3.1 Dataset Summary
- **Dataset:** MNIST Handwritten Digit Database
- **Primary Hosted Links:**
  - **Hugging Face (Official Yann LeCun Hub):** [https://huggingface.co/datasets/ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist)
  - **Papers With Code Benchmark:** [https://paperswithcode.com/dataset/mnist](https://paperswithcode.com/dataset/mnist)
  - **PyTorch S3 Binary Mirror:** [https://ossci-datasets.s3.amazonaws.com/mnist/](https://ossci-datasets.s3.amazonaws.com/mnist/)
  - **Original Academic Reference:** Yann LeCun, Corinna Cortes, Christopher J.C. Burges (1998) [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/) *(Note: NYU server uses plain HTTP; HTTPS triggers SSL handshake errors, which is why PyTorch downloads from the AWS S3 mirror).*
- **Library Loader:** `torchvision.datasets.MNIST` (downloaded to `notebooks/data/`)
- **Dimensions:** $28 \times 28$ grayscale images (single channel)
- **Number of Classes:** 10 categorical digit classes (0 through 9)

### 3.2 Dataset Partitioning (Preventing Data Leakage)
To prevent data leakage and ensure reproducible evaluation, the dataset is strictly partitioned into three independent subsets:

| Subset | Image Count | Purpose | Partitioning Method |
|:---|---:|:---|:---|
| **Training** | 54,000 | Model parameter updates | Split from official 60k train split with `Generator(seed=42)` |
| **Validation** | 6,000 | Hyperparameter check & model checkpoint selection | Split from official 60k train split with `Generator(seed=42)` |
| **Test** | 10,000 | Final held-out evaluation & robustness benchmarking | Official, untouched MNIST test set |

### 3.3 Preprocessing Decisions (Deliverable 4 Requirements)
In accordance with course Deliverable 4, all preprocessing transformations are documented and justified:
1. **Scaling & Normalization:** Images are converted to floating-point tensors using `transforms.ToTensor()`, scaling raw 8-bit integer pixel intensities $[0, 255]$ into $[0.0, 1.0]$. Z-score standardization (subtracting dataset mean and dividing by standard deviation) is deliberately omitted to preserve the physical pixel interval $[0.0, 1.0]$ required for valid adversarial perturbation clipping.
2. **Missingness & Outlier Handling:** MNIST contains zero missing values, zero unreadable headers, and zero corrupted arrays. Data imputation or outlier filtering is unnecessary.
3. **Encoding:** Ground truth labels are categorical integers ($y \in \{0, 1, \dots, 9\}$), which map directly into PyTorch's `nn.CrossEntropyLoss`. One-hot encoding is unnecessary.
4. **Class Imbalance Handling:** As established in our Exploratory Data Analysis, the training set is well-balanced across all 10 classes (~5,400 to ~6,700 images per class, ~9%–11% each). Synthetic oversampling (SMOTE) and class-weighted loss functions are unnecessary.

---

## 4. Methodology & Architecture

### 4.1 Model Architecture (`MNISTCNN`)
Both baseline and robust classifiers share the identical compact Convolutional Neural Network architecture:
- **Feature Extractor:**
  - `Conv2d(1, 16, kernel_size=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (feature map: $16 \times 14 \times 14$)
  - `Conv2d(16, 32, kernel_size=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2, 2)` (feature map: $32 \times 7 \times 7$)
- **Classifier Head:**
  - `Flatten()` $\rightarrow$ `Linear(1568, 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(128, 10)`
- **Total Trainable Parameters:** 206,922 parameters (CPU-friendly, reproducible in under 2 minutes).

### 4.2 Fast Gradient Sign Method (FGSM)
FGSM is a single-step, gradient-based white-box adversarial attack that perturbs input pixels in the direction that maximizes classification loss:

$$\boldsymbol{x}_{\text{adv}} = \text{clip}\Big(\boldsymbol{x} + \epsilon \cdot \text{sign}\big(\nabla_{\boldsymbol{x}} \mathcal{L}(\boldsymbol{\theta}, \boldsymbol{x}, y)\big), 0, 1\Big)$$

Where:
- $\boldsymbol{x}$ is the clean input image.
- $y$ is the true class label.
- $\boldsymbol{\theta}$ represents the model parameters.
- $\mathcal{L}$ is the cross-entropy loss.
- $\epsilon$ is the perturbation magnitude controlling attack budget.
- The perturbed image is projected back into valid pixel space using $[0, 1]$ clipping.

### 4.3 Training Protocols
- **Baseline Training:** Optimized over 5 epochs using clean images, Adam optimizer ($\text{lr}=0.001$), batch size 128, cross-entropy loss, and random seed 42. Checkpoint with highest validation accuracy is saved to `models/baseline_cnn.pth`.
- **Adversarial Training:** Optimized over 5 epochs where every training mini-batch is transformed on-the-fly into FGSM adversarial examples at $\epsilon_{\text{train}} = 0.10$ using the active network gradients. Model weights are saved to `models/adversarial_cnn.pth`.

---

## 5. Experimental Results & Robustness Evaluation

Both models were evaluated on the complete 10,000-image test set across five perturbation strengths: $\epsilon \in \{0.00, 0.05, 0.10, 0.15, 0.20\}$.

### 5.1 Test Classification Accuracy vs. Perturbation Strength

| Perturbation ($\epsilon$) | Baseline CNN Accuracy | Adversarially Trained CNN Accuracy | Absolute Difference |
|---:|---:|---:|---:|
| **0.00 (Clean)** | **98.80%** | **98.75%** | -0.05% |
| **0.05** | 94.82% | 97.63% | +2.81% |
| **0.10** | 83.07% | 95.80% | +12.73% |
| **0.15** | 60.98% | 92.62% | +31.64% |
| **0.20** | 34.63% | 87.05% | **+52.42%** |

### 5.2 Attack Success Rate (ASR)
Attack Success Rate measures the proportion of samples that were originally classified correctly on clean data but subsequently misclassified under the adversarial attack:

$$\text{ASR}(\epsilon) = \frac{\sum_{i=1}^N \mathbf{1}\big(f(\boldsymbol{x}_i) = y_i \land f(\boldsymbol{x}_i^{\text{adv}}) \neq y_i\big)}{\sum_{i=1}^N \mathbf{1}\big(f(\boldsymbol{x}_i) = y_i\big)} \times 100\%$$

| Perturbation ($\epsilon$) | Baseline CNN ASR | Adversarially Trained CNN ASR | Relative Reduction |
|---:|---:|---:|---:|
| **0.00** | 0.00% | 0.00% | 0.00% |
| **0.05** | 4.03% | 1.13% | 71.96% reduction |
| **0.10** | 15.92% | 2.99% | 81.22% reduction |
| **0.15** | 38.28% | 6.21% | 83.78% reduction |
| **0.20** | 64.95% | 11.85% | **81.76% reduction** |

### 5.3 Quantitative Analysis & Interpretation
1. **Clean Accuracy Preservation:** The adversarially trained CNN achieves **98.75%** clean test accuracy, matching the baseline CNN (98.80%) within 0.05%. This confirms that for MNIST, robustness does not incur an excessive accuracy tax on clean distributions.
2. **Robustness Generalization:** While the baseline model collapses to **34.63%** accuracy at $\epsilon = 0.20$, the robust model maintains **87.05%** accuracy—a **+52.42 percentage point advantage**.
3. **Attack Deflection:** At maximum tested perturbation ($\epsilon = 0.20$), the attack success rate drops from **64.95%** on the baseline down to **11.85%** on the robust model, demonstrating substantial resilience against single-step gradient attacks.

---

## 6. Generated Visualizations

All generated visualizations are saved in the `results/` directory:
- **`results/model_accuracy_comparison.png`:** Robust-accuracy comparison curve across $\epsilon \in [0.0, 0.20]$.
- **`results/attack_success_comparison.png`:** Attack Success Rate comparison curve across $\epsilon \in [0.0, 0.20]$.
- **`results/adversarial_examples.png`:** Side-by-side visual panels displaying Clean images, Baseline FGSM attacked images, and Robust FGSM attacked images with explicit outcomes (**Attack Succeeded** vs. **Attack Defended**).
- **`results/class_distribution.png`:** Class balance bar chart across all 10 digit classes.
- **`results/sample_digits.png`:** Visual sample grid showing handwriting diversity.
- **`results/average_digits.png`:** Archetypal mean digit images illustrating core stroke invariance and geometric overlap.
- **`results/model_comparison.csv`:** Full quantitative evaluation metrics table.

---

## 7. How to Run & Reproduce Experiments

### Step 1: Environment Setup
Ensure Python 3.10+ is installed. Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/your-username/Adversarial-Training-for-Model-Robustness.git
cd Adversarial-Training-for-Model-Robustness

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### Step 2: Run Exploratory Data Analysis (EDA)
Launch Jupyter Notebook to inspect or re-run the annotated EDA notebook:

```bash
jupyter notebook notebooks/01_mnist_eda.ipynb
```

### Step 3: Run Model Tests (Smoke Tests)
Verify that model architectures, shapes, and attack functions work properly:

```bash
python src/test_model.py
python src/test_attack.py
```

### Step 4: Train Baseline and Robust Models
Train both models from scratch (pre-trained checkpoints are also included in `models/`):

```bash
# Train clean baseline CNN
python src/train.py

# Train adversarially robust CNN (FGSM eps=0.10)
python src/adversarial_train.py
```

### Step 5: Evaluate Robustness Across Attack Strengths
Run the benchmark evaluation across all epsilon levels ($\epsilon \in \{0.0, 0.05, 0.10, 0.15, 0.20\}$) to generate `results/model_comparison.csv` and evaluation curves:

```bash
python src/evaluate.py
```

### Step 6: Generate Attack Visualizations
Generate side-by-side visual examples of successful and defended attacks:

```bash
python src/visualize_attacks.py
```

*Note: All scripts can be executed directly as shown above (e.g., `python src/evaluate.py`) or as Python modules (e.g., `python -m src.evaluate`).*

---

## 8. Limitations & Future Work

1. **Dataset Simplicity:** MNIST images feature high-contrast, segmented characters against zero-intensity backgrounds. Robustness dynamics may differ on complex natural image benchmarks such as CIFAR-10 or ImageNet.
2. **Single-Step vs. Multi-Step Attacks:** The defense is trained against single-step FGSM attacks. While highly effective against FGSM, single-step adversarial training can suffer from gradient masking, remaining vulnerable to iterative attacks like Projected Gradient Descent (PGD) or Carlini-Wagner (C&W).
3. **White-Box Evaluation:** This evaluation tested white-box attacks where perturbations are derived directly from the target model's gradients. Future extensions could evaluate black-box transfer attacks across disparate architectures.

---

## 9. Deliverables Compliance Checklist

| Deliverable | Requirement | Repository Implementation |
|---|---|---|
| **1. Problem Definition** | Objectives, research questions, motivation, background | README Sections 1 & 2; literature background |
| **2. Dataset Documentation** | Source, counts, features, target, limitations | README Section 3; `notebooks/01_mnist_eda.ipynb` |
| **3. EDA** | $\ge 3$ exploratory visualizations with interpretation | `sample_digits.png`, `class_distribution.png`, `average_digits.png` in `notebooks/01_mnist_eda.ipynb` |
| **4. Preprocessing** | Document cleaning, scaling, splitting, imbalance justification | README Section 3.3; `src/data.py` (54k/6k/10k, seed 42) |
| **5. Baseline & Method** | Working baseline and proposed method with parameters | `src/model.py`, `src/train.py`, `src/adversarial_train.py` |
| **6. Experiments** | $\ge 3$ meaningful configurations/comparisons | 10 evaluations (Baseline vs. Robust across 5 $\epsilon$ levels) |
| **7. Evaluation** | Results table and $\ge 3$ result visualizations | `model_comparison.csv`, `model_accuracy_comparison.png`, `attack_success_comparison.png`, `adversarial_examples.png` |
| **8. Analysis & Conclusion** | Explain trade-offs, failure cases, limitations, conclusions | README Sections 5.3, 5.4, 8, 9 |
| **9. Reproducible Code** | Notebooks, source code, seeds, README run instructions | Complete code in `src/`, seeds=42, `requirements.txt`, README Section 7 |
| **10. Report & Demo** | Structured report and presentation | To be completed by team for final presentation |

---

## 10. References

1. **MNIST Benchmark:** LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). *The MNIST Database of Handwritten Digits*. [https://yann.lecun.com/exdb/mnist/](https://yann.lecun.com/exdb/mnist/)
2. **FGSM Formulation:** Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. International Conference on Learning Representations (ICLR). [https://arxiv.org/abs/1412.6572](https://arxiv.org/abs/1412.6572)
3. **Adversarial Training Concepts:** Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR. [https://arxiv.org/abs/1706.06083](https://arxiv.org/abs/1706.06083)
4. **PyTorch Framework:** Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS.
