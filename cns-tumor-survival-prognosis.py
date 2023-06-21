# -*- coding: utf-8 -*-

# Install required python packages
# These were not coded by the author
!pip install scikit-survival

# Commented out IPython magic to ensure Python compatibility.
# Import required python libraries
# These are available libraries and were not coded by the author


import pandas as pd
import numpy as np
import math
from tabulate import tabulate
import matplotlib.pyplot as plt
import sklearn
import sklearn.metrics as metrics
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import train_test_split
import sksurv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder, encode_categorical
from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Read and inspect clinical dataset
df = pd.read_csv('/content/drive/MyDrive/REMBRANDT/rembrandt_clinical.csv')
df.head(5)

# Clean up dataframe
df['survival_months'] = df.apply(lambda row: row.survival_days/30, axis=1)
df.rename(columns={'deceased': 'vital_status'}, inplace=True)

df.head(2)

# Drop unnecessary columns (fields)
df.drop(['sample', 'survival_days'], axis=1, inplace=True)

# Check for missing or NULL values
df.isnull().any()

# Split the targets from the features
data_x = df.iloc[:,0:-2]
data_y = df.iloc[:,-2:]
data_status = df.iloc[:,-2:-1].to_numpy()
data_survival = df.iloc[:,-1:].to_numpy()


# Generate Kaplan-Meier curve for the entire dataset
time, survival_prob = sksurv.nonparametric.kaplan_meier_estimator(data_y['vital_status'], data_y['survival_months'])
plt.step(time, survival_prob, where="post")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.title("Survival probability based on time for every patient")

# Check distinct KPS scores in dataset
data_x["karnofsky"].value_counts()

# Generate Kaplan-Meier curve for different KPS scores
for kps_score in (80, 90, 100):
    mask_treat = data_x["karnofsky"] == kps_score
    time_kps, survival_prob_kps = kaplan_meier_estimator(
        data_y["vital_status"][mask_treat],
        data_y["survival_months"][mask_treat])

    plt.step(time_kps, survival_prob_kps, where="post",
             label="KPS Score = %s" % kps_score)

plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")
plt.title("Survival probability based on KPS Scores")

# Generate Kaplan-Meier curve for different types of radiation therapy
for value in data_x["rad_type"].unique():
    mask = data_x["rad_type"] == value
    time_cell, survival_prob_cell = kaplan_meier_estimator(data_y["vital_status"][mask],
                                                           data_y["survival_months"][mask])
    plt.step(time_cell, survival_prob_cell, where="post",
             label="%s (n = %d)" % (value, mask.sum()))

plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")

# Generate Kaplan-Meier curve for different types of chemotherapy
for value in data_x["chemo"].unique():
    mask = data_x["chemo"] == value
    time_cell, survival_prob_cell = kaplan_meier_estimator(data_y["vital_status"][mask],
                                                           data_y["survival_months"][mask])
    plt.step(time_cell, survival_prob_cell, where="post",
             label="%s (n = %d)" % (value, mask.sum()))

plt.title("Survival probability based on KPS Scores")
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")

# Check datatype of each field
data_x.dtypes

# Make a working copy of dataframe and convert all the Object types to Category type
data_x_copy = data_x.copy()
data_x_copy[data_x_copy.select_dtypes(['object']).columns] = data_x_copy.select_dtypes(['object']).apply(lambda x: x.astype('category'))

# One-hot encoding to convert Category type to Numeric type
data_x_numeric = data_x_copy.copy()
cat_cols = data_x_numeric.select_dtypes(['category']).columns
cat_to_code = {col: dict(zip(data_x_numeric[col], data_x_numeric[col].cat.codes)) for col in cat_cols}
code_to_cat = {k: {v2: k2 for k2, v2 in v.items()} for k, v in cat_to_code.items()}
#code_to_cat
data_x_numeric[cat_cols] = data_x_numeric[cat_cols].apply(lambda x: x.cat.codes)
data_x_numeric = data_x_numeric.replace(np.nan, 0, regex=True)

# Convert target to a structured numpy array
arr = np.array(data_y, dtype=object)

# Create the target to have both the Vital Status and Survival Months
# This is the required format of the target variables
dt=dtype=[('vital_status', '?'), ('survival_months', '<f8')]
struct_data_y=np.array([tuple(row) for row in arr], dtype=dt)

# Fit a Cox Proportional Hazards model to the data
set_config(display="text")  # displays text representation of estimators

estimator = CoxPHSurvivalAnalysis()
estimator.fit(data_x_numeric, struct_data_y)

# Generate Log hazard ratio
pd.Series(estimator.coef_, index=data_x_numeric.columns)

# Cox Net
estimator = sksurv.linear_model.CoxnetSurvivalAnalysis(fit_baseline_model=True)
estimator.fit(data_x_numeric, struct_data_y)

# Create new "test" patients to predict survival
# Think of this as the input from 5 patients whose survival we want the model to predict
# Fields - age_at_dx,	gender,	disease,	grade,	race,	karnofsky,	steroid,	anti-convulsant,	rad_type,	chemo,	surgery
x_new = pd.DataFrame.from_dict({
    1: [4, 0, 0, 1, 2, 80, 0, 0, 0, 0, 0],
    2: [8, 1, 1, 0, 2, 80, 6, 2, 2, 4, 2],
    3: [7, 0, 2, 3, 1, 90, 3, 4, 3, 6, 3],
    4: [6, 1, 3, 2, 1, 100, 0, 2, 2, 4, 4],
    5: [2, 0, 2, 2, 1, 80, 4, 3, 1, 3, 2]},
     columns=data_x_numeric.columns, orient='index')

# Run CoxPH model prediction on test patients
pred_surv = estimator.predict_survival_function(x_new)
time_points = np.arange(1, 100)
for i, surv_func in enumerate(pred_surv):
    plt.step(time_points, surv_func(time_points), where="post",
             label="Patient %d" % (i + 1))
plt.ylabel("Survival Probability $\hat{S}(t)$")
plt.xlabel("Survival Time $t$ in Months")
plt.legend(loc="best")


pred_surv = estimator.predict_survival_function(data_x_numeric)
time_points = np.arange(1, 251)
for i, surv_func in enumerate(pred_surv):
    plt.step(time_points, surv_func(time_points), where="post",
             label="Sample %d" % (i + 1))
plt.ylabel("Survival Probability $\hat{S}(t)$")
plt.xlabel("Survival Time $t$ in Months")
plt.legend(loc="upper right")

# One-hot encode categorical columns
cat_cols = data_x_numeric.select_dtypes(['category']).columns
cat_to_code = {col: dict(zip(data_x_numeric[col], data_x_numeric[col].cat.codes)) for col in cat_cols}
code_to_cat = {k: {v2: k2 for k2, v2 in v.items()} for k, v in cat_to_code.items()}

#code_to_cat
x_new[cat_cols] = x_new[cat_cols].apply(lambda x : x.k)
x_new.head(2)

# Generate the Concordance Index score


prediction = estimator.predict(data_x_numeric)
result = concordance_index_censored(data_y["vital_status"], data_y["survival_months"], prediction)
result[0]

# Assess the importance/impact of each variable on prediction

def fit_and_score_features(X, y):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    m = CoxPHSurvivalAnalysis()
    for j in range(n_features):
        Xj = X[:, j:j+1]
        m.fit(Xj, y)
        scores[j] = m.score(Xj, y)
    return scores

scores = fit_and_score_features(data_x_numeric.values, struct_data_y)
pd.Series(scores, index=data_x_numeric.columns).sort_values(ascending=False)

# Construct pipeline to run k-fold validation
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

pipe = Pipeline([('select', SelectKBest(fit_and_score_features, k=3)),
                 ('model', CoxPHSurvivalAnalysis())])

# Run the k-fold validation pipeline to select the most impactful group of variables that offer highest prediction accuracy

from sklearn.model_selection import GridSearchCV, KFold

param_grid = {'select__k': np.arange(1, data_x_numeric.shape[1] + 1)}
cv = KFold(n_splits=3, random_state=1, shuffle=True)
gcv = GridSearchCV(pipe, param_grid, return_train_score=True, cv=cv)
gcv.fit(data_x_numeric, struct_data_y)

results = pd.DataFrame(gcv.cv_results_).sort_values(by='mean_test_score', ascending=False)
results.loc[:, ~results.columns.str.endswith("_time")]

# Fit model using the automatically generated most significant independent variables
pipe.set_params(**gcv.best_params_)
pipe.fit(data_x_numeric, struct_data_y)

encoder, transformer, final_estimator = [s[1] for s in pipe.steps]
pd.Series(final_estimator.coef_, index=encoder.encoded_columns_[transformer.get_support()])

x_train, x_test, y_train, y_test = train_test_split(data_x_numeric, struct_data_y, test_size=0.2, stratify=struct_data_y["vital_status"], random_state=0)

cph = make_pipeline(CoxPHSurvivalAnalysis())
cph.fit(x_train, y_train)

# Generate time dependent risk score for Cox PH model
times = np.arange(2, 98, 7)
cph_risk_scores = cph.predict(x_test)
cph_auc, cph_mean_auc = cumulative_dynamic_auc(
    y_train, y_test, cph_risk_scores, times
)

plt.plot(times, cph_auc, marker="o")
plt.axhline(cph_mean_auc, linestyle="--")
plt.xlabel("days from enrollment")
plt.ylabel("time-dependent AUC")
plt.grid(True)

# Time dependent risk scores for Random survival forest

rsf = make_pipeline(RandomSurvivalForest(n_estimators=100, min_samples_leaf=7, random_state=0))
rsf.fit(x_train, y_train)

rsf_chf_funcs = rsf.predict_cumulative_hazard_function(
    x_test, return_array=False)
rsf_risk_scores = np.row_stack([chf(times) for chf in rsf_chf_funcs])

rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
    y_train, y_test, rsf_risk_scores, times
)

score_cindex = pd.Series(
    [
        rsf.score(x_test, y_test),
        cph.score(x_test, y_test),
        0.5,
    ],
    index=["RSF", "CPH", "Random"], name="c-index",
)

score_cindex.round(3)

rsf_surv_prob = np.row_stack([
    fn(times)
    for fn in rsf.predict_survival_function(x_test)
])

cph_surv_prob = np.row_stack([
    fn(times)
    for fn in cph.predict_survival_function(x_test)
])

km_func = StepFunction(
    *kaplan_meier_estimator(y_test["vital_status"], y_test["survival_months"])
)
km_surv_prob = np.tile(km_func(times), (y_test.shape[0], 1))

random_surv_prob = 0.5 * np.ones(
    (y_test.shape[0], times.shape[0])
)

score_brier = pd.Series(
    [
        integrated_brier_score(struct_data_y, y_test, prob, times)
        for prob in (rsf_surv_prob, cph_surv_prob, random_surv_prob, km_surv_prob)
    ],
    index=["RSF", "CPH", "Random", "Kaplan-Meier"],
    name="IBS"
)

pd.concat((score_cindex, score_brier), axis=1).round(3)