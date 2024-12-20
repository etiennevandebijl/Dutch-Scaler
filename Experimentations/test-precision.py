import DutchScaler

import numpy as np

# %% Precision Check, Precision is very sensitive to amounts of M and P

P = 1000
N = 10

y_true = [1] * P + [0] * N

DDB = DutchScaler.lower_bound(y_true, "PPV")
print(DDB)

# score = 0.899

# alpha, _ = DutchScaler.optimized_indicator_inverted(y_true, "PPV", score)
# print(alpha)


scores =  np.linspace(DDB, 1.0)
alphas = []
for score in scores:
    
    alpha, _ = DutchScaler.optimized_indicator_inverted(y_true, "PPV", score)
    alphas.append(alpha)

alphas
import matplotlib.pyplot as plt

plt.figure(figsize = (10,10))
plt.plot(scores, alphas)
plt.show()