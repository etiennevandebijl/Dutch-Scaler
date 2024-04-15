import numpy as np
import pandas as pd

import DutchDraw as DutchDraw
import DutchScaler as DutchScaler

#%% Check Function 1

#Excluded: G2
metric_options = ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "TS", "FM"]

results = []

for M in [10, 30, 50]:
    for P in range(1, M):
        y_true = [1] * P + [0] * (M - P)

        for metric in metric_options:
            upper_rho = DutchScaler.valid_rho_values(y_true, metric)
            for rho in np.linspace(0, upper_rho, 5)[:-1]:
                baseline = DutchDraw.optimized_baseline_statistics(y_true, metric)['Max Expected Value']    
                upper_limit = DutchScaler.upper_bound(y_true, metric, rho)
                for s in np.linspace(baseline, upper_limit): 
                    alpha, thetaopts = DutchScaler.optimized_indicator_inverted(y_true, metric, s, rho)
                    score_v1 = DutchScaler.indicator_score(y_true, metric, alpha, thetaopts, rho)
                    results.append([metric, M, P, baseline, upper_limit, rho, alpha, s, score_v1])
        
df = pd.DataFrame(results, columns = ["Metric", "M", "P", "Baseline", "Upper Bound", "rho", "alpha", "Score", "Score_v1"])

df["Diff v1"] = np.abs(df["Score"] - df["Score_v1"])

#%% Check Function 1 and 2

#Excluded: G2, MCC, MK
metric_options = ["PPV", "NPV", "ACC", "BACC", "FBETA", "J", "KAPPA", "TS", "FM"]

results = []

for M in [10, 30, 50]:
    for P in range(1, M):
        y_true = [1] * P + [0] * (M - P)
        for metric in metric_options:
            upper_rho = DutchScaler.valid_rho_values(y_true, metric)
            for rho in np.linspace(0, upper_rho, 5)[:-1]:
                baseline = DutchDraw.optimized_baseline_statistics(y_true, metric)['Max Expected Value']    
                upper_limit =  DutchScaler.upper_bound(y_true, metric, rho)
                for s in np.linspace(baseline, upper_limit): 
                    alpha, thetaopts = DutchScaler.optimized_indicator_inverted(y_true, metric, s, rho)
                    score_v1 = DutchScaler.indicator_score(y_true, metric, alpha, thetaopts, rho)
                    score_v2 = DutchScaler.optimized_indicator(y_true, metric, alpha, rho)
                    results.append([metric, M, P, baseline, upper_limit, alpha, s, score_v1, score_v2])
        
df = pd.DataFrame(results, columns = ["Metric", "M", "P", "Baseline", "Upper Bound", "alpha", "Score", "Score_v1", "Score_v2"])

df["Diff v1"] = np.abs(df["Score"] - df["Score_v1"])
df["Diff v2"] = np.abs(df["Score"] - df["Score_v2"])