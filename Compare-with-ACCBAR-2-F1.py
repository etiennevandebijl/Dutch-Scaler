#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:29:33 2023

@author: etienne
"""

import pandas as pd
import DutchScaler as DutchScaler
import DutchDraw as DutchDraw
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/home/etienne/Dropbox/Projects/Dutch Scaler/Data/Table-ACCBAR-article-f1.csv")

df.columns = ["N", "P", "F1"]

#df.loc[len(df)] = {'N': 1000, 'P': 1, 'F1': 0.1}

results = []
baselines = []
for index, row in df.iterrows():
    y_true = [1] * int(row["P"]) + [0] * int(row["N"])
    bs = DutchDraw.DutchDraw_baseline(y_true, "FBETA")["Max Expected Value"]
    baselines.append(bs)
    alpha, _ = DutchScaler.optimized_indicator_inverted(y_true, "FBETA", row["F1"])
    results.append(alpha)

df["Baseline"] = baselines
df["ACCBAR (Delta)"] = df["F1"] - baselines
df["DSPI (Alpha)"] = results

df['Accuracy Ranked'] = df["F1"].rank(ascending=False)
df['ACCBAR (Delta) Ranked'] = df["ACCBAR (Delta)"].rank(ascending=False)
df['DSPI (Alpha) Ranked'] = df['DSPI (Alpha)'].rank(ascending=False)

#df = df.sort_values('Accuracy')
df["Study"] = list(range(df.shape[0]))



df = df[["Study", "F1", "Baseline", "ACCBAR (Delta)", "DSPI (Alpha)", "ACCBAR (Delta) Ranked", "DSPI (Alpha) Ranked"]]


#df.plot(x="Study", y=["Baseline", "ACCBAR (Delta)", "DSPI (Alpha)"], kind="bar", rot=0, figsize = (12,6), title = "Accuracy = Baseline + Delta")

df["ACCBAR (Delta) Scaled"] = (df["F1"] - df["Baseline"]) / (1 - df["Baseline"])

df.plot(x="Study", y=["ACCBAR (Delta) Scaled", "DSPI (Alpha)", "Baseline", "F1"], kind="bar", rot=0, figsize = (12,6))



# %%
columns_to_plot = [ ["Baseline", "ACCBAR (Delta)"], "ACCBAR (Delta) Scaled", "DSPI (Alpha)"]

fig, ax = plt.subplots(figsize=(10, 6))

bar_spots = len(columns_to_plot)
bar_width = 0.8 / bar_spots

pos = np.arange(len(df))
dodge_offsets = np.linspace(- bar_spots * bar_width / 2, bar_spots * bar_width / 2, bar_spots, endpoint = False)
for columns, offset in zip(columns_to_plot, dodge_offsets):
    bottom = 0
    for col in ([columns] if isinstance(columns, str) else columns):
        ax.bar(pos + offset, df[col], bottom=bottom, width=bar_width, align='edge', label=col)
        bottom += df[col]
ax.set_xticks(pos)
ax.set_xticklabels(df['Study'], rotation=0)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
plt.ylabel("Score")
plt.xlabel("Study")
plt.tight_layout()
#plt.title("F1 = Baseline + Delta")
plt.show()


# %%

