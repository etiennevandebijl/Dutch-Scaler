import numpy as np
import DutchDraw as DutchDraw
import pandas as pd
import matplotlib.pyplot as plt

measure_dictionary = DutchDraw.measure_dictionary

def learning_curves(alpha, measure, M, P, beta = 1):
    N = M - P
    
    if measure in measure_dictionary['J']:
        return alpha
    
    if measure in measure_dictionary['BACC']:
        return 0.5 * alpha + 0.5
    
    if measure in measure_dictionary['ACC']:
        return alpha * (min(N, P) / M) + max(P, N) / M
    
    if measure in measure_dictionary['PPV']:
        up = alpha * P * (M - 1) + P
        down = alpha * M * (P - 1) + M
        return up/down

    if measure in measure_dictionary['NPV']:
        up = alpha * N * (M - 1) + N
        down = alpha * M * (N - 1) + M
        return up/down

    if measure in measure_dictionary['FM']:
        up = np.sqrt(P)
        down = np.sqrt(alpha * -N + M)
        return up / down

    if measure in measure_dictionary['FBETA']:
        up = (1 + beta * beta) * P
        down = alpha * -N + M + P * beta * beta
        return up / down

M = 30
P = 10

metrics = ["PPV", "NPV", "ACC", "BACC", "FBETA", "J","FM"]

results = []
for alpha in np.linspace(0, 1):
    for b in metrics:
        score = learning_curves(alpha, b, M, P, beta = 1)
        results.append([alpha, b, score])
        
results = []
for alpha in np.linspace(0, 1):
    for b in [0.001, 0.1, 0.5, 1,2,3,4,5,6,7,8,9,20,50, 10000, np.inf]:
        score = learning_curves(alpha, "FBETA", M, P, beta = b)
        results.append([alpha, b, score])
        
df = pd.DataFrame(results, columns = ["Alpha", "Metric", "Score"])

plt.figure(figsize = (10,10))
for metric, group, in df.groupby("Metric"):
    plt.plot(group["Alpha"], group["Score"], label = metric)
plt.xlabel("Alpha")
plt.ylabel("Score")
plt.legend()
plt.show()

plt.figure(figsize = (10,10))
for metric, group, in df.groupby("Metric"):
    group["Scaled-Score"] = (group["Score"] - group["Score"].min()) / (1 - group["Score"].min())
    plt.plot(group["Alpha"], group["Scaled-Score"], label = metric)
plt.xlabel("Alpha")
plt.ylabel("(Score - DDB) / (1 - DDB)")
plt.legend()
plt.show()