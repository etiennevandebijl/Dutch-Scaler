import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DutchDraw as DutchDraw
from DSPI_inverse import DSPI_inverse
# %% Experiment 1
'''
This is some old code. Rho is not integrated here. Also for MK we have 
shown that only 1 solution holds. 
'''

M = 500
P = 23
theta = 0.2
score = 0.9
theta_star = round(theta * M) / M

def calculate_MK(a, t, m, p, n, mk):
    PPV_d = (a * p * (1 - t) + p * t)
    PPV_n = (a * (p - m * t) + m * t)
    PPV = PPV_d/PPV_n
    NPV_d = (a * n * t + n * (1 - t))
    NPV_n = (a * (m*t - p) + m * (1 - t))
    NPV = NPV_d / NPV_n
    return PPV + NPV - 1 - mk

def calculate_alpha(t, m, p, n, mk):
    a = (p - m * t) * (p - m * t) * mk
    b = (p - m * t) * (n - m * (mk + 1)) + m * p * (1 - t) + 2 * (p - m * t) * m * t * mk
    c = m * m * t * mk * (t - 1)
    
    a_1 = (-b + np.sqrt(b*b - 4 * a * c) ) / (2 * a)
    a_2 = (-b - np.sqrt(b*b - 4 * a * c) ) / (2 * a)
    return a_1, a_2

a_1, a_2 = calculate_alpha(theta_star, M, P, M - P, score)
print(a_1)
print(a_2)

result = calculate_MK(a_1, theta_star, M, P, M-P, score)
print(result)

# %% Experiment 2
'''
This is checked and integrated in the code.
'''
results = []
for m in [10, 30, 50, 100, 150]:
    for p in range(1, m):
        for s in [0.1, 0.2, 0.4, 0.5, 0.6, 0.9]:
            for theta in [0.1, 0.2, 0.4, 0.5, 0.6, 0.9]:
                t_ = round(theta * m) / m
                if True: #(p - m * t_) != 0:
                    a_1, a_2 = calculate_alpha(t_, m, p, m-p, s)
                    result = calculate_MK(a_1, t_, m, p, m-p, s)
                    results.append([m, p, s, t_, a_1, result])

df = pd.DataFrame(results, columns = ["M", "P", "MK","Theta*", "a_1","Result"])

# p - m * t mag geen 0 zijn

# %% Experiment 3
'''
This code was to check how the Dutch Scaler compares with a simple min-max scaler.
'''

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

# %% Experiment 4
'''
This piece of code shows that the G2 DDB is below the mu_0.
'''
M = 30
P = 1

y_true = [1] * P + [0] * (M - P)

baseline = DutchDraw.optimized_baseline_statistics(y_true, "G2")

print(baseline["Max Expected Value"]) #0.39148014634313555
print(baseline["Argmax Expected Value"]) #0.6666666666666666

tres = baseline["Argmax Expected Value"][0]

results = []
for t in [tres]:
    for alpha in np.linspace(0,1):
        TPR = alpha * (1 - t) + t
        TNR = alpha * t + 1 - t
        G2DO = np.sqrt(TPR * TNR)
        
        results.append([alpha, t, G2DO])

df = pd.DataFrame(results, columns = ["Alpha", "Theta", "G2DO"])

plt.figure(figsize = (10, 10))
for theta, group in df.groupby("Theta"):
    plt.plot(group["Alpha"], group["G2DO"], label = round(theta,2))
plt.axhline(baseline["Max Expected Value"], label = "E[G^2_theta]", color = "red")
plt.xlabel("Alpha")
plt.ylabel("G^2_alpha")
plt.legend()
plt.show()

# %% Experiment 5
'''
The idea behind this experiment was to test is identical alpha's for all metrics would be given for a single prediction.
This was not the case.
'''

metrics = ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "TS", "G2"]

for m in [30]:
    for p in [5]:
        y_true = [1] * p + [0] * (m - p)
        random_flip = np.random.rand(m)
        y_pred = [1 - x if z < 0.05 else x for x, z in zip(y_true, random_flip)]
        
        tp = DutchDraw.measure_score(y_true, y_pred, "TP")
        tn = DutchDraw.measure_score(y_true, y_pred, "TN")
        fp = DutchDraw.measure_score(y_true, y_pred, "FP")
        fn = DutchDraw.measure_score(y_true, y_pred, "FN")
        results = {"tp": [tp, "*", p, 0, 1], "tn": [tn, "*", 0, (m - p), 0]}
        results["fn"] = [fn, "*", 0, 0, 1]
        results["fp"] = [fp, "*", 0, 0, 0]
        for metric in metrics:
            sm = DutchDraw.measure_score(y_true, y_pred, metric)
            alpha, thetaopts = DSPI_inverse(y_true, metric, sm)
            TPa = alpha * p + (1 - alpha) * p * thetaopts[0]
            TNa = alpha * (m - p) * thetaopts[0] + (1 - thetaopts[0]) * (m - p)
            results[metric] = [sm, alpha, TPa, TNa, thetaopts[0]]
            
df = pd.DataFrame(results).T
df.columns = ["Score", "Alpha", "TPa", "TNa", "Thetaopt"]
print(df)
