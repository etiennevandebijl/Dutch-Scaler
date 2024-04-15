import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DutchDraw as DutchDraw
import DutchScaler as DutchScaler

# %% Plot 1 Experiment different Beta values

# Settings
M = 100 
P = 10
rho = 0.0

metric = "FBETA"

y_true = [1] * P + [0] * (M - P)

results = []
for alpha in np.linspace(0, 1):
    for b in [0.001, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 10000, np.inf]:
        score = DutchScaler.optimized_indicator(y_true, metric, alpha, rho, beta = b)
        results.append([alpha, b, score])
        
df = pd.DataFrame(results, columns = ["Alpha", "Metric", "Score"])

plt.figure(figsize = (10,10))
for metric, group, in df.groupby("Metric"):
    plt.plot(group["Alpha"], group["Score"], label = metric)
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$F_\beta$')
plt.legend()
plt.show()

# %% Plot 2 Scaler

M = 100 
P = 10
rho = 0.0

metric_options = ["ACC", "BACC", "FBETA",  "FM", "G2",  "J", "KAPPA", "MCC", "MK", "NPV", "PPV",     "TS"]
#metric_options = ["BACC", "J", "PPV", "NPV"]

y_true = [1] * P + [0] * (M - P)

results = []
for metric in metric_options:
    baseline = DutchDraw.optimized_baseline_statistics(y_true, metric)['Max Expected Value']    
    upper_limit = DutchScaler.upper_bound(y_true, metric, rho)
    for s in np.linspace(baseline, upper_limit): 
        alpha, thetaopts = DutchScaler.optimized_indicator_inverted(y_true, metric, s, rho)
        score_v1 = DutchScaler.indicator_score(y_true, metric, alpha, thetaopts, rho)
        results.append([metric, M, P, baseline, upper_limit, rho, alpha, s, score_v1])

df = pd.DataFrame(results, columns = ["Metric", "M", "P", "Baseline", "Upper Bound", "rho", "Alpha", "Score", "Score_v1"])

plt.figure(figsize = (10,10))
for metric, group, in df.groupby("Metric"):
    plt.plot(group["Alpha"], group["Score"], label = metric)
plt.xlabel(r'$\alpha$')
#plt.ylabel(r"DSPI")
plt.ylabel(r"$\mu_{\alpha}$")
plt.ylim(0,1)
plt.legend()
plt.show()

# %% Plot 3 Scaler Scaled Directly

M = 100 
P = 10
rho = 0.0

metric_options = ["PPV", "NPV", "ACC", "BACC", "FBETA", "J", "KAPPA", "FM", "TS"]

y_true = [1] * P + [0] * (M - P)

results = []
for metric in metric_options:
    print(metric)
    UB = DutchScaler.upper_bound(y_true, metric, rho)
    LB = DutchScaler.lower_bound(y_true, metric)
    for alpha in np.linspace(0,1): 
        score = DutchScaler.optimized_indicator(y_true, metric, alpha, rho)
        score_scaled = (score - LB) / (UB - LB)
        results.append([metric, M, P, rho, alpha, score_scaled])

df = pd.DataFrame(results, columns = ["Metric", "M", "P", "rho", "Alpha", "Score"])

plt.figure(figsize = (10,10))
for metric, group, in df.groupby("Metric"):
    plt.plot(group["Alpha"], group["Score"], label = metric)
plt.ylabel(r'$\frac{\mu_\alpha - \mu_0}{\mu_1 - \mu_0}$')
plt.xlabel(r'$\alpha$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

# %% Plot 4 Scaler Scaled Indirectly

M = 100
P = 10
rho = 0.05

metric_options = ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "TS", "FM"]
#metric_options = ["FBETA"]

y_true = [1] * P + [0] * (M - P)

results = []
for metric in metric_options:
    UB = DutchScaler.upper_bound(y_true, metric, rho)
    LB = DutchScaler.lower_bound(y_true, metric)
    for s in np.linspace(LB, UB): 
        alpha, thetaopts = DutchScaler.optimized_indicator_inverted(y_true, metric, s, rho)
        score = DutchScaler.indicator_score(y_true, metric, alpha, thetaopts, rho)
        score_scaled = (score - LB) / (UB - LB)
        results.append([metric, M, P, rho, alpha, score_scaled])

df = pd.DataFrame(results, columns = ["Metric", "M", "P", "rho", "Alpha", "Score"])

plt.figure(figsize = (10,10))
for metric, group, in df.groupby("Metric"):
    plt.plot(group["Alpha"], group["Score"], label = metric)
plt.ylabel(r'$\frac{\mu_\alpha - \mu_0}{\mu_1 - \mu_0}$')
plt.xlabel(r'$\alpha$')
plt.legend()
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

# %% Plot 5 Effect Rho

M = 10 
P = 4

metric = "MCC"

y_true = [1] * P + [0] * (M - P)

results = []

upper_rho = DutchScaler.valid_rho_values(y_true, metric)
for rho in np.linspace(0, upper_rho, 10)[:-1]:
    UB = DutchScaler.upper_bound(y_true, metric, rho)
    LB = DutchScaler.lower_bound(y_true, metric)
    for alpha in np.linspace(0,1): 
        score = DutchScaler.optimized_indicator(y_true, metric, alpha, rho)
        score_scaled = (score - LB) / (UB - LB)
        results.append([M, P, rho, alpha, score, score_scaled])

df = pd.DataFrame(results, columns = ["M", "P", "rho", "Alpha", "Score", "Scaled-Score"])

plt.figure(figsize = (10,10))
for r, group, in df.groupby("rho"):
    plt.plot(group["Alpha"], group["Score"], label = r)
plt.ylabel(r'$\mu_\alpha$')
plt.xlabel(r'$\alpha$')
plt.legend()
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()


plt.figure(figsize = (10,10))
for r, group, in df.groupby("rho"):
    plt.plot(group["Alpha"], group["Scaled-Score"], label = r)
plt.ylabel(r'$\frac{\mu_\alpha - \mu_0}{\mu_1 - \mu_0}$')
plt.xlabel(r'$\alpha$')
plt.legend()
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()



