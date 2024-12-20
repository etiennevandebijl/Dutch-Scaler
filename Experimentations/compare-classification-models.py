from DutchDraw import  DutchDraw
import DutchScaler
import pandas as pd
import numpy as np

METRICS = ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "TS"]

# %% Data
# Machine learning in medicine: a practical introduction
model_1 = ["GLM", 148, 10, 2, 67]
model_2 = ["SVM", 146, 5, 4, 72]
model_3 = ["ANN", 148, 11, 2 ,66]
combined = [model_1, model_2, model_3]

df = pd.DataFrame(combined, columns = ["Model","TN", "FN", "FP", "TP"])
df["P"] = df["TP"] + df["FN"]
df["N"] = df["TN"] + df["FP"]

# %%

results = []
for index, row in df.iterrows():

    y_true = [1] * int(row["P"]) + [0] * int(row["N"])
        
    y_pred = [1] * int(row["TP"]) + [0] * int(row["FN"]) + [1] * int(row["FP"]) + [0] * int(row["TN"])

    mu_row = {**row, "Score":"mu"}
    mu_a_row = {**row,  "Score":"mua"}
    accbar_row = {**row,  "Score":"accbar"}

    
    for metric in METRICS:

        bs = DutchDraw.DutchDraw_baseline(y_true, metric)["Max Expected Value"]
        
        score = DutchDraw.measure_score(y_true, y_pred, metric, beta=1)
    
        alpha, _ = DutchScaler.optimized_indicator_inverted(y_true, metric, score)
        
        mu_row[metric] = round(score,3)
        mu_a_row[metric] = round(alpha,3)
        accbar_row[metric] = round((score - bs) / (1 - bs), 3) 
        
    results.append(mu_row)
    results.append(mu_a_row)
    results.append(accbar_row)
        
df_results = pd.DataFrame(results)
df_results["Average"] = round(df_results[METRICS].mean(1),3)
df_results = df_results.T

# %% 

# https://ieeexplore-ieee-org.vu-nl.idm.oclc.org/stamp/stamp.jsp?arnumber=8681044
# 1541 keer geciteerd.

path = "/home/etienne/Dropbox/Projects/The Dutch Scaler Performance Indicator How much did my model actually learn/Data/"

df_data = pd.read_csv(path + "Data-Stats-Deep-Learning-Approach-for-Intelligent-IDS.csv")
df_classifier_results = pd.read_csv(path + "Results-BC-Deep-Learning-Approach-for-Intelligent-IDS.csv")

df_comb = pd.merge(df_classifier_results, df_data, on = ["Dataset"], how = "left")

results = []
for index, row in df_comb.iterrows():

    y_true = [1] * int(row["P"]) + [0] * int(row["N"])
        
    new_row = {**row}
    
    for abbr, metric in {"ACC":"Accuracy", "FBETA": "F1", "PPV": "Precision"}.items():
    
        score =  row[metric]
        if score < DutchScaler.lower_bound(y_true, abbr):
            alpha = np.nan
        else:
            alpha, _ = DutchScaler.optimized_indicator_inverted(y_true, abbr, score)

        new_row[metric] = score
        if metric == "Precision":
            new_row[metric + "-DS"] = round(alpha,5)
        else:
            new_row[metric + "-DS"] = round(alpha,3)
    results.append(new_row)
        
df_results = pd.DataFrame(results)

df_results.drop(["M","P", "N"], axis = 1, inplace = True)
df_results = df_results[df_results["Dataset"].isin(["NSL-KDD"])]
df_results["Average-DS"] = round(df_results[[a for a in df_results.columns if "-DS" in a]].mean(1),3)

#%% Heart Disease


path = "/home/etienne/Dropbox/Projects/The Dutch Scaler Performance Indicator How much did my model actually learn/Data/"

df_data = pd.read_csv(path + "heart-disease-detection-data.csv")

results = []
for index, row in df_data.iterrows():

    y_true = [1] * int(row["P"]) + [0] * int(row["N"])
        
    new_row = {**row}
    
    for abbr, metric in {"ACC":"Accuracy", "FBETA": "F-Score", "PPV": "Precision"}.items():
    
        score =  row[metric]
        if score < DutchScaler.lower_bound(y_true, abbr):
            alpha = np.nan
        else:
            alpha, _ = DutchScaler.optimized_indicator_inverted(y_true, abbr, score)

        new_row[metric] = score
        if metric == "Precision":
            new_row[metric + "-DS"] = round(alpha,5)
        else:
            new_row[metric + "-DS"] = round(alpha,3)
    results.append(new_row)
        
df_results = pd.DataFrame(results)

df_results.drop(["M","P", "N"], axis = 1, inplace = True)
df_results["Average-DS"] = round(df_results[[a for a in df_results.columns if "-DS" in a]].mean(1),3)




