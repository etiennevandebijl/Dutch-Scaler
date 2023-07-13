
import DutchDraw as DutchDraw
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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