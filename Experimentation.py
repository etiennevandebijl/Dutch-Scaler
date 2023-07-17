from DSPI import DSPI
from DSPI_inverse import DSPI_inverse
import DutchDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

metrics = ["ACC","BACC", "FBETA", "FM", "NPV", "MCC", "KAPPA", "PPV", "TS", "MK", "J"]

results = []
for m in [40]:
    for p in [9]:
        y_true = [1] * p + [0] * (m - p)
        for metric in metrics:
            baseline = DutchDraw.optimized_baseline_statistics(y_true, metric)['Max Expected Value']       
            for s in np.linspace(baseline, 1): 
                
                alpha, thetaopts = DSPI_inverse(y_true, metric, s)
                reverse_score = DSPI(y_true, metric, alpha, thetaopts)

                results.append([m, p, s, metric, alpha, thetaopts, np.abs(s - reverse_score), baseline])

df = pd.DataFrame(results, columns = ["M", "P", "Score DO", "Metric", "Alpha", "Thetaopts", "Reverse_score", "Baseline"])


#%% 


# def derivative_fbeta(M, P, alpha, beta):
#     N = M - P
#     d = N * (1 + beta * beta) * P
#     n = (alpha * -N + M + P * beta**2)**2
#     return d /n

# def fbeta(M, P, alpha, beta):
#     N = M - P
#     d = (1 + beta * beta) * P
#     n = alpha * -N + M + P * beta * beta
#     return d/n

# alpha_ex = 0.75
# der = derivative_fbeta(m, p, alpha_ex, 1)
# FBaR =  fbeta(m, p, alpha_ex, 1)
# b = FBaR - der *alpha_ex 

# y_1 = der * (alpha_ex - 0.2) + b
# y_2 = der * (alpha_ex + 0.2) + b


# x = np.linspace(0, 1)
# y = [derivative_fbeta(m, p, i, 1) for i in x]
# y_bla = [1 for i in x]
# plt.figure(figsize = (10,10))
# plt.plot(x, y)e)
# plt.plot(x, y_bla)
# plt.xlabel("Alpha")
# plt.show()

plt.figure(figsize = (8,8))
for metric, group in df.groupby("Metric"):
    #plt.plot(group["Alpha"], group["Score DO"], label = metric)
    plt.plot(group["Score DO"], group["Alpha"], label = metric)
 #   plt.plot([alpha_ex - 0.2, alpha_ex + 0.2], [y_1, y_2], label = "Derivative")
plt.legend()
# plt.ylabel(r'$\mu_\alpha$')
plt.xlabel(r"$\overline{\mu}$")
# plt.xlabel(r'$\alpha$')
plt.ylabel(r"DSPI")
plt.ylim(0,1)
plt.show()

plt.figure(figsize = (8,8))
for metric, group in df.groupby("Metric"):
    plt.plot(group["Alpha"], group["Score DO"], label = metric)
plt.legend()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\mu_{\alpha}$")
plt.xlim(0,1)

plt.show()

plt.figure(figsize = (8,8))
for metric, group in df.groupby("Metric"):
    group["Scaled-Score"] = (group["Score DO"] - group["Baseline"].mean()) / (1 - group["Baseline"].mean())
    plt.plot(group["Alpha"], group["Scaled-Score"], label = metric)


plt.legend()
plt.ylabel(r'$\frac{\mu_\alpha - \mu_0}{\mu_1 - \mu_0}$')
plt.xlabel(r'$\alpha$')
#plt.title("Minimal Alpha required to achieve realised score M = 30, P = 10")
plt.xlim(0,1)
plt.show()



# plt.figure(figsize = (10,10))
# for metric, group in df.groupby("Metric"):
#     group["Scaled-Score"] = np.sqrt(((group["Score"] - group["Baseline"].mean()) / (1 - group["Baseline"].mean()) - group["Alpha"])**2)
#     plt.plot(group["Alpha"], group["Scaled-Score"], label = metric)
# plt.legend()
# plt.ylabel("|(Score - DDB) / (1 - DDB) - alpha|")
# plt.xlabel("Alpha")
# plt.title("Minimal Alpha required to achieve realised score M = 30, P = 4")
# plt.show()



# =============================================================================
# Hier een bewijs dat niet altijd alle alpha's hetzelfde zijn 
# =============================================================================

# metrics = ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "TS", "G2"]

# for m in [30]:
#     for p in [5]:
#         y_true = [1] * p + [0] * (m - p)
#         random_flip = np.random.rand(m)
#         y_pred = [1 - x if z < 0.05 else x for x, z in zip(y_true, random_flip)]
        
#         tp = DutchDraw.measure_score(y_true, y_pred, "TP")
#         tn = DutchDraw.measure_score(y_true, y_pred, "TN")
#         fp = DutchDraw.measure_score(y_true, y_pred, "FP")
#         fn = DutchDraw.measure_score(y_true, y_pred, "FN")
#         results = {"tp": [tp, "*", p, 0, 1], "tn": [tn, "*", 0, (m - p), 0]}
#         results["fn"] = [fn, "*", 0, 0, 1]
#         results["fp"] = [fp, "*", 0, 0, 0]
#         for metric in metrics:
#             sm = DutchDraw.measure_score(y_true, y_pred, metric)
#             alpha, thetaopts = dutch_oracle_determine_alpha(y_true, metric, sm)
#             TPa = alpha * p + (1 - alpha) * p * thetaopts[0]
#             TNa = alpha * (m - p) * thetaopts[0] + (1 - thetaopts[0]) * (m - p)
#             results[metric] = [sm, alpha, TPa, TNa, thetaopts[0]]
            
# df = pd.DataFrame(results).T
# df.columns = ["Score", "Alpha", "TPa", "TNa", "Thetaopt"]
# print(df)

