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
df = pd.read_csv("/home/etienne/Dropbox/Projects/The Dutch Scaler Performance Indicator How much did my model actually learn/Data/Survey_Classification_Performance_Evaluation_And_Reporting_78_Studies_Table_4_Overview.csv")

# %%  Determine Dutch Draw Baseline and the Dutch Scaler Performance Indicator
 
dutchscaler_alphas = []
baselines = []
for index, row in df.iterrows():
    y_true = [1] * int(row["P"]) + [0] * int(row["N"])
    bs = DutchDraw.DutchDraw_baseline(y_true, "Accuracy")["Max Expected Value"]
    baselines.append(bs)
    alpha, _ = DutchScaler.optimized_indicator_inverted(y_true, "Accuracy", row["Accuracy"])
    dutchscaler_alphas.append(alpha)

df["Baseline"] = baselines
df["ACCBAR (Delta)"] = df["Accuracy"] - baselines
df["DSPI (Alpha)"] = dutchscaler_alphas

df['Accuracy Ranked'] = df["Accuracy"].rank(ascending=False)
df['ACCBAR (Delta) Ranked'] = df["ACCBAR (Delta)"].rank(ascending=False)
df['DSPI (Alpha) Ranked'] = df['DSPI (Alpha)'].rank(ascending=False)
df["ACCBAR (Delta) Scaled"] = (df["Accuracy"] - df["Baseline"]) / (1 - df["Baseline"])


# %% Visualize performance

columns_to_plot = [ ["Baseline", "ACCBAR (Delta)"], "ACCBAR (Delta) Scaled", "DSPI (Alpha)" ]

fig, ax = plt.subplots(figsize=(10, 5))

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
ax.set_xticklabels(df['Nr'], rotation=0)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
plt.ylabel("Score")
plt.xlabel("Study")
plt.tight_layout()
plt.show()

