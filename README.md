# Machine Learning Pipeline for Central Nervous System Tumor Diagnosis and Outcome Prediction

* Research selected as a finalist in the International Science and Engineering Fair (Translational Medicine Category) and won the American Statistical Association Special Award. 
* National Congresssional App Challenge winner; invited to the Capitol in D.C. to present research. 
* Presented at the Junior Humanities and Science Symposium ($1500 award) and American Junior Academy of Sciences Conference. 
* Presented at Sigma Xi Forum on International Excellence.

This repository contains code for developing and evaluating machine learning models for diagnosis and prognosis of central nervous system (CNS) tumors using multi-modal clinical and imaging-derived features.

The goal of this project is to build robust, interpretable models that support clinical decision-making in high-risk neurological disease settings, where accurate prediction of tumor type, grade, and patient outcomes directly impacts treatment planning and survival.

---
# Algorithm Structure
<img width="903" height="366" alt="Screenshot 2026-02-27 at 2 00 41 PM" src="https://github.com/user-attachments/assets/8045c33e-eec9-44a1-a415-346afce2655a" />

---
# Repository Structure
| Path | Description |
|------|-------------|
| `cns-tumor-classification-diagnosis.py/` | End-to-end pipeline for CNS tumor diagnosis classification. Includes data preprocessing, model training, evaluation (accuracy, ROC-AUC, etc.), and performance reporting. |
| `cns-tumor-survival-prognosis.py/` | Survival modeling pipeline for prognosis prediction. Implements survival analysis methods (e.g., Cox proportional hazards), computes concordance index (C-index), and generates survival risk outputs. |

## 🔬 Techniques Used

This repository implements both **supervised classification** and **survival analysis** methods for CNS tumor diagnosis and prognosis using structured clinical and imaging-derived features.

---

### 1. Diagnostic Classification

The classification pipeline predicts tumor diagnosis or grade using supervised learning techniques.

#### Models Implemented

**Logistic Regression**

A linear probabilistic classifier:

\[
P(Y=1 \mid X) = \sigma(\beta^T X)
\]

Where:
- \( \sigma \) is the logistic (sigmoid) function  
- \( \beta \) represents learned feature coefficients  

Logistic regression provides:
- Interpretable coefficients
- Directionality of feature effects
- A strong linear baseline

---

**Random Forest Classifier**

An ensemble of decision trees:

\[
\hat{y} = \frac{1}{B} \sum_{b=1}^{B} T_b(X)
\]

Where:
- \(T_b\) is an individual decision tree  
- \(B\) is the number of trees  

Random forests:
- Capture nonlinear relationships  
- Model complex feature interactions  
- Reduce variance via ensembling  

---

#### Preprocessing Techniques

- Standardization of continuous features  
- Handling of missing values  
- Train/test split for validation  

---

#### Classification Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC–AUC**
- **Confusion Matrix**

These metrics evaluate overall performance and class-specific behavior.

---

### 2. Survival Prognosis Modeling

The prognosis pipeline predicts time-to-event outcomes such as overall survival.

#### Cox Proportional Hazards Model

A semi-parametric survival model:

\[
h(t \mid X) = h_0(t)\exp(\beta^T X)
\]

Where:
- \(h(t \mid X)\) = hazard function  
- \(h_0(t)\) = baseline hazard  
- \(\beta\) = covariate coefficients  

The model estimates:
- Hazard ratios  
- Relative risk scores  
- Survival probability estimates  

---

#### Survival Evaluation Metric

**Concordance Index (C-index)**

\[
C = P(\hat{T}_i < \hat{T}_j \mid T_i < T_j)
\]

The C-index measures how well predicted risk ordering matches actual survival ordering.

---

### 📊 Model Validation Strategy

- Train/test split  
- Cross-validation (where applicable)  
- Out-of-sample evaluation  
- Comparison of linear vs ensemble methods  

---

### 🔍 Interpretability

- Logistic regression coefficients  
- Random forest feature importance  
- Hazard ratios from Cox models  

These outputs support clinically interpretable conclusions about which features influence diagnosis and survival risk.
