# Comparative Analysis of Focal Loss vs. Binary Cross-Entropy in Linear Models

This repository contains an empirical evaluation framework designed to analyze the behavior, mathematical edge cases, and performance tradeoffs of **Focal Loss** relative to classic **Binary Cross-Entropy (BCE)** when applied to rigid linear models across 49 imbalanced tabular datasets.

---

## Project Overview
Class imbalance presents a classic challenge in machine learning, where dominant majority classes often overwhelm loss functions during optimization. This project explores whether **Focal Loss**—originally engineered by Lin et al. (2017) to solve severe foreground-background pixel imbalances in complex deep learning architectures (RetinaNet)—can provide a geometric or statistical advantage when constrained entirely within a **Logistic Regression** classifier.

The repository is structured into two main experimental phases:
1. **`no_focal_loss.ipynb` (Phase 1 Baseline):** Custom implementation of Logistic Regression optimized via standard, unweighted Binary Cross-Entropy.
2. **`with_focal_loss.ipynb` (Phase 2 Focal Loss):** Custom implementation extending the model to optimize an exotic Focal Loss function with automated asymmetric inverse-frequency class weighting ($\alpha$-balancing).

---

## Mathematical Architecture

Both models utilize a custom, object-oriented framework optimized from scratch using first-order Gradient Descent. To compute precise gradients for non-standard loss setups, the optimization layer utilizes the `autograd` library for seamless automatic differentiation.

### 1. Binary Cross-Entropy Baseline
The standard unweighted loss treating all classification errors uniformly:
$$BCE(p) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

### 2. Custom Focal Loss Formulation
The objective function breaks error symmetry by introducing a modulating parameter $(1 - p_t)^\gamma$:
$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

* **Focusing Parameter ($\gamma = 2.0$):** Dynamically downweights the loss contribution of easily classified examples ($p > 0.7$), forcing the model to allocate its gradient updates to hard, ambiguous, or rare instances.
* **Alpha Weighting ($\alpha_t$):** Automatically assigns class weights inversely proportional to target frequencies to account for severe dataset skewness.

Both architectures incorporate stable sigmoid probability mapping ($\sigma(z) = \frac{1}{1 + e^{-z}}$), programmatic clipping ($10^{-12}$) to prevent logarithmic underflow, and an adjustable $L_2$ Tikhonov regularization penalty.

---

## Evaluation Pipeline
To test performance across varying degrees of class skewness, the benchmarking loop evaluates models against 49 distinct tabular datasets using an automated data processing engine:
* **Feature Processing:** Automatic text string tokenization, median-imputation for missing values, and rigorous Z-score scaling (`StandardScaler`).
* **Validation:** $5$-Fold Stratified Cross-Validation (`StratifiedKFold`) to preserve the exact minority class density across all optimization splits.
* **Resilient Metrics:** Models are benchmarked using metrics strictly resilient to class imbalance: $\text{ROC-AUC}$, $\text{PR-AUC}$, $\text{F1-Score}$, $\text{Recall}$, and $\text{Specificity}$.

---

## Key Findings & Post-Mortem Analysis
The benchmarking routine revealed an unexpected but critical machine learning design lesson: **the advanced Focal Loss algorithm consistently degraded performance compared to the classic unweighted Cross-Entropy baseline.**

### Global Averages Across All 49 Datasets:
| Optimization Strategy | Global $\text{ROC-AUC}$ | Global $\text{PR-AUC}$ |
| :--- | :---: | :---: |
| **Phase 1: Binary Cross-Entropy Baseline** | **0.7064** | **0.5712** |
| **Phase 2: Custom Focal Loss ($\gamma=2.0, \alpha$)** | **0.5571** | **0.4402** |

### Analytical Breakdown of the Mismatch:
1. **The Core Linearity Constraint:** Focal loss changes the *magnitude* of weight updates during training, but it cannot expand structural model capacity. A Logistic Regression classifier can only ever learn a completely flat hyper-plane decision boundary. When datasets present complex, non-linear feature overlaps, changing the loss formulation cannot resolve geometric intersection.
2. **Gradient Compression & Stalled Convergence:** The focal factor $(1-p)^\gamma$ suppresses the gradients of confident predictions. When utilizing first-order gradient descent with fixed steps over short iteration windows ($500$ iterations), these micro-gradients severely stall convergence, causing the weights to settle before establishing a stable boundary.
3. **The Alpha-Balancing Over-Correction Loop:** Concurrently introducing explicit inverse-frequency scaling ($\alpha$) alongside structural focal penalties causes massive optimization bias. The linear model over-corrects to escape punishing majority-class loss flags, tanking precision metrics without yielding any significant boost in true recall.

---
## Getting Started

### Prerequisites
Ensure your local environment includes the required mathematical and array processing libraries:
```bash
pip install numpy pandas scikit-learn autograd tqdm
\`\`\`

### Reproducing the Experiments
1. Configure your local file paths in both notebooks to match your local copy of the target data directory:
   ```python
   DATA_DIR = Path(r"path/to/your/class_imbalance_folder")
