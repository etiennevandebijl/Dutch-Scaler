import pandas as pd
import DutchScaler as DutchScaler
import DutchDraw as DutchDraw
import matplotlib.pyplot as plt

metric = "ACC"

if metric == "ACC":
    df = pd.read_csv("/home/etienne/Dropbox/Projects/Dutch Scaler/Data/Table-ACCBAR-article-2.csv")
else:
    df = pd.read_csv("/home/etienne/Dropbox/Projects/Dutch Scaler/Data/Table-ACCBAR-article-f1.csv")

df.columns = ["N", "P", "Accuracy"]

results = []
baselines = []
for index, row in df.iterrows():
    y_true = [1] * int(row["P"]) + [0] * int(row["N"])
    bs = DutchDraw.DutchDraw_baseline(y_true, metric)["Max Expected Value"]
    baselines.append(bs)
    alpha, _ = DutchScaler.optimized_indicator_inverted(y_true, metric, row[metric])
    results.append(alpha)

df["Baseline"] = baselines
df["Delta"] = df[metric] - baselines
df["Dutch Scaler"] = results

df["Ratio"] = df["P"] / df["N"]

df['Accuracy'] = df[metric].rank(ascending=False)
df['ACCBAR'] = df["Delta"].rank(ascending=False)
df['DutchScaler'] = df['Dutch Scaler'].rank(ascending=False)

df = df.sort_values('Accuracy').reset_index()
df["Study"] = list(range(df.shape[0]))



df = df[["Study", "ACC", "Delta", "ACCBAR", "Dutch Scaler"]]


plt.figure(figsize = (8,8))
df[['Accuracy','ACCBAR','DutchScaler']].plot.bar(figsize = (8,6))
plt.ylabel("Rank")
plt.xticks([], [])
plt.show()


df.plot.scatter(x = "Delta", y = "Dutch Scaler", figsize = (6,6))
