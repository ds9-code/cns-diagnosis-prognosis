!pip install torchtuples

!pip install pycox

import torch # For building the networks 
import torchtuples as tt # Some useful functions

from pycox.models import LogisticHazard
# from pycox.models import PMF
# from pycox.models import DeepHitSingle
from pycox.evaluation import EvalSurv

# Pycox format requires the survival months (time) first, and the vital status (event) next
data_y = data_y[data_y.columns[::-1]]
data_y.head()

dframes = [data_x_numeric, data_y]
df_pycox = pd.concat(dframes, axis=1)
df_pycox["vital_status"] = df_pycox["vital_status"].astype(int)

df_pycox.head()

df_train = df_pycox.copy()
df_test = df_train.sample(frac=0.2)
df_train = df_train.drop(df_test.index)
df_val = df_train.sample(frac=0.2)
df_train = df_train.drop(df_val.index)

df_train.head()

# For preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

cols_standardize = []
cols_leave = df_train.columns.values.tolist()

standardize = [([col], StandardScaler()) for col in cols_standardize]
leave = [(col, None) for col in cols_leave]

x_mapper = DataFrameMapper(standardize + leave)

x_train = x_mapper.fit_transform(df_train).astype('float32')
x_val = x_mapper.transform(df_val).astype('float32')
x_test = x_mapper.transform(df_test).astype('float32')

df_pycox_float = df_pycox.astype('float32')

num_durations = 10

labtrans = LogisticHazard.label_transform(num_durations)
# labtrans = PMF.label_transform(num_durations)
# labtrans = DeepHitSingle.label_transform(num_durations)

get_target = lambda df: (df['survival_months'].values, df['vital_status'].values)
y_train = labtrans.fit_transform(*get_target(df_train))
y_val = labtrans.transform(*get_target(df_val))

train = (x_train, y_train)
val = (x_val, y_val)

# We don't need to transform the test labels
durations_test, events_test = get_target(df_test)

labtrans.cuts

y_train

labtrans.cuts[y_train[0]]

in_features = x_train.shape[1]
num_nodes = [32, 32]
out_features = labtrans.out_features
batch_norm = True
dropout = 0.1

net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)

model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)

batch_size = 256
epochs = 100
callbacks = [tt.cb.EarlyStopping()]

log = model.fit(x_train, y_train, batch_size, epochs, callbacks, val_data=val)

_ = log.plot()

log.to_pandas().val_loss.min()

model.score_in_batches(val)

surv = model.predict_surv_df(x_test)

surv.iloc[:, :5].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

surv = model.interpolate(10).predict_surv_df(x_test)

surv.iloc[:, :5].plot(drawstyle='steps-post')
plt.ylabel('S(t | x)')
_ = plt.xlabel('Time')

ev = EvalSurv(surv, durations_test, events_test, censor_surv='km')

ev.concordance_td('antolini')

time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
ev.brier_score(time_grid).plot()
plt.ylabel('Brier score')
_ = plt.xlabel('Time')

ev.nbll(time_grid).plot()
plt.ylabel('NBLL')
_ = plt.xlabel('Time')

ev.integrated_brier_score(time_grid)

plt.plot(times, cph_auc, "o-", label="CoxPH (mean AUC = {:.3f})".format(cph_mean_auc))
plt.plot(times, rsf_auc, "o-", label="RSF (mean AUC = {:.3f})".format(rsf_mean_auc))
plt.plot(times, surv_auc, "o-", label="Deep Survnet (mean AUC = {:.3f})".format(surv_mean_auc))


plt.xlabel("Months from Diagnosis")
plt.ylabel("Time-dependent AUC")
plt.legend(loc="lower right")
plt.grid(True)