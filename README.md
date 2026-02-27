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

Logistic regression models the probability of class membership using:

P(Y = 1 | X) = sigmoid(beta^T X)

Where:
- beta represents learned feature weights  
- sigmoid(z) = 1 / (1 + e^(−z))

This provides:
- Interpretable coefficients
- Directionality of feature effects
- A strong linear baseline

**Random Forest Classifier**

Random forest is an ensemble of decision trees:

Prediction = (1 / B) * sum of tree_b(X) over B trees

Where:
- B = number of trees  
- Each tree is trained on bootstrapped samples  

Random forests:
- Capture nonlinear relationships  
- Model feature interactions  
- Reduce overfitting via ensembling  

### 2. Survival Prognosis Modeling

The prognosis pipeline predicts time-to-event outcomes such as overall survival.

#### Cox Proportional Hazards Model

The Cox model estimates the hazard function:

h(t | X) = h0(t) * exp(beta^T X)

Where:
- h(t | X) is the hazard at time t  
- h0(t) is the baseline hazard  
- exp(beta^T X) represents relative risk  

Outputs include:
- Hazard ratios (exp(beta))  
- Risk scores  
- Survival probability estimates  

#### Survival Evaluation Metric

**Concordance Index (C-index)**

Concordance Index (C-index):

C = Probability(predicted risk ordering matches true survival ordering)

The C-index measures how well predicted risk rankings align with observed survival times.
