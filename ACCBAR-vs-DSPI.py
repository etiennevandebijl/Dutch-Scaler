import DutchScaler as DutchScaler
import DutchDraw as DutchDraw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


for M in [5, 50, 100, 200, 500]:
    for P in [1, int(M/2), M - 1]:

        N = M - P
        y_true = [1] * P + [0] * N
        bs = DutchDraw.DutchDraw_baseline(y_true, "FBETA")["Max Expected Value"]

        results = []

        for f1 in np.linspace(bs, 1.0):
            alpha, _ = DutchScaler.optimized_indicator_inverted(y_true, "FBETA", f1)
            results.append([f1, alpha])

        df = pd.DataFrame(results, columns = ["F1", "DSPI"])

        df["ACCBAR (Delta)"] = df["F1"] - bs
        df["ACCBAR (Delta) Scaled"] = (df["F1"] - bs) / (1 - bs)

        plt.figure(figsize = (10,10))
        plt.plot(df["F1"], df["DSPI"], label = "DSPI")
        plt.plot(df["F1"], df["ACCBAR (Delta) Scaled"], label = " ACCBAR (Delta) Scaled")
        plt.xlabel("F1")
        plt.ylabel("Score")
        plt.legend()
        plt.title("ACCBAR vs DSPI with P=" + str(P) + " and M =" + str(M))
        plt.show()
        
# %% Stel P = 1


results = [] 
for M in [5, 100, 1000, 5000, 10000]:
    N = M - 1
    y_true = [1] * P + [0] * N
    bs = DutchDraw.DutchDraw_baseline(y_true, "FBETA")["Max Expected Value"]

    for f1 in np.linspace(bs, 1.0):
        alpha, _ = DutchScaler.optimized_indicator_inverted(y_true, "FBETA", f1)
        
        results.append([f1, alpha, M, (f1 - bs) / (1 - bs)])

df = pd.DataFrame(results, columns = ["F1", "DSPI", "M", "ACCBAR (Delta) Scaled"])

plt.figure(figsize = (10,10))

for m, group in df.groupby("M"):
    plt.plot(group["F1"], group["DSPI"], label = "DSPI M = " + str(m))
    plt.plot(group["F1"], group["ACCBAR (Delta) Scaled"], label = "ACCBAR M = " + str(m))
plt.xlabel("F1")
plt.ylabel("Score")
plt.legend()
plt.title("ACCBAR versus DSPI with P=1")
plt.show()