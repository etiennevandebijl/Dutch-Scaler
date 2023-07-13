

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N = 20
P = 10

ddb_f1 = 2 * P / (2* P + N)
do_f1 = 1

def min_max_scaler(score, ddb_f1):
    a = (score - ddb_f1) / (1 - ddb_f1)
    return a

a = min_max_scaler(0.75, ddb_f1)

def do_scaler(score, P, N):
    a = 1 - (2 * P * (1 - score) / (N * score))
    return a

results = []
for sc in np.linspace(ddb_f1,1,100):
    a_1 = min_max_scaler(sc, ddb_f1)
    a_2 = do_scaler(sc, P, N)
    results.append([sc, a_1, a_2])
    

df = pd.DataFrame(results, columns = ["score", "min-max","DO"])

plt.figure()
plt.plot(df["min-max"], df["score"], label = "Min-Max scaler")
plt.plot(df["DO"], df["score"], label = "DO")
plt.xlabel("Alpha")
plt.ylabel("F1 score")
plt.legend()
plt.show()

#%% BACC


ddb_Bacc = 0.5
do_bacc = 1

def min_max_scaler(score, ddb_f1):
    a = (score - ddb_f1) / (1 - ddb_f1)
    return a

a = min_max_scaler(0.75, ddb_f1)

def do_scaler_bacc(score, P, N):
    a = 2 * score - 1
    return a

results = []
for sc in np.linspace(ddb_f1,1,100):
    a_1 = min_max_scaler(sc, ddb_f1)
    a_2 = do_scaler_bacc(sc, P, N)
    results.append([sc, a_1, a_2])
    

df = pd.DataFrame(results, columns = ["score", "min-max","DO"])

plt.figure()
plt.plot(df["min-max"], df["score"], label = "Min-Max scaler")
plt.plot(df["DO"], df["score"], label = "DO")
plt.xlabel("Alpha")
plt.ylabel("BACC score")
plt.legend()
plt.show()