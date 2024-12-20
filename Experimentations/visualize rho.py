import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DutchDraw as DutchDraw
import DutchScaler as DutchScaler

# %% Plot 1 Experiment different Beta values

# Settings
M = 100 
P = 10

y_true = [1] * P + [0] * (M - P)

results = []
metric_options = ["ACC", "BACC", "FBETA",  "FM", "J", "KAPPA", "MK", "NPV", "PPV",  "G2", "TS"]

for metric in metric_options:
    
    baseline = DutchDraw.optimized_baseline_statistics(y_true, metric)['Max Expected Value']    
 
    for score in np.linspace(baseline, 1.0): 
        print(metric)
        rho = DutchScaler.select_rho(y_true, metric, score)

        results.append([rho, metric, score])
df = pd.DataFrame(results, columns = ["rho", "Metric", "Score"])


#%% 

plt.figure(figsize = (10,7))
for metric, group, in df.groupby("Metric"):
    plt.plot(group["rho"], group["Score"], label = metric)
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\mu_1$')
plt.legend()
plt.show()


DutchScaler.select_rho(y_true, "TS", 0.9)
