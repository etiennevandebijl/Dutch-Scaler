import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DutchScaler as DutchScaler
import DutchDraw as DutchDraw

#%% Load data

# Citation for data: 
'''
Gürol Canbek, Tugba Taskata Temizel, and Seref Sagiroglu, “Binary-Classification Performance Evaluation Reporting Survey Data with the Findings”, 
Mendeley Data, v3, 2020, http://dx.doi.org/10.17632/5c442vbjzg.3
'''
df = pd.read_csv("/home/etienne/Dropbox/Projects/The Dutch Scaler Performance Indicator How much did my model actually learn/Data/Survey_Classification_Performance_Evaluation_And_Reporting_78_Studies_Table_4_Overview_Subset_F1.csv")

# %% Determine Dutch Draw Baseline and the Dutch Scaler Performance Indicator

dutchscaler_alphas = []
baselines = []
for index, row in df.iterrows():
    y_true = [1] * int(row["P"]) + [0] * int(row["N"])
    bs = DutchDraw.DutchDraw_baseline(y_true, "FBETA")["Max Expected Value"]
    baselines.append(bs)
    alpha, _ = DutchScaler.optimized_indicator_inverted(y_true, "FBETA", row["F1"])
    dutchscaler_alphas.append(alpha)

df["DD baseline"] = baselines
df["ACCBAR (Delta)"] = df["F1"] - baselines
df["DSPI (Alpha)"] = dutchscaler_alphas

df['Accuracy Ranked'] = df["F1"].rank(ascending=False)
df['ACCBAR (Delta) Ranked'] = df["ACCBAR (Delta)"].rank(ascending=False)
df['DSPI (Alpha) Ranked'] = df['DSPI (Alpha)'].rank(ascending=False)
df["ACCBAR (Delta) Scaled"] = (df["F1"] - df["DD baseline"]) / (1 - df["DD baseline"])

df = df.sort_values("Accuracy Ranked")

# %%
columns_to_plot = [ ["DD baseline", "ACCBAR (Delta)"], "ACCBAR (Delta) Scaled", "DSPI (Alpha)"]

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

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
plt.ylabel("Score")


# Idea 1
# ax.set_xticklabels(df['Nr'], rotation=0)
# plt.xlabel("Study")


ax.set_xticklabels(np.arange(1, df.shape[0] + 1), rotation=0)
plt.xlabel(r"Study (Ranked by $F_1$)")

plt.tight_layout()
plt.savefig("/home/etienne/Dropbox/Projects/The Dutch Scaler Performance Indicator How much did my model actually learn/Results/DSPI-ACCBAR on f1 - dissertation.png")

plt.show()

