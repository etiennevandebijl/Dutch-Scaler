import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import DutchDraw as DutchDraw
import DutchScaler as DutchScaler


METRICS = ["ACC", "BACC", "FBETA", "FM", "G2", "J", "KAPPA", "MCC", "MK", "NPV", "PPV", "TS"]
#colors = plt.cm.get_cmap('tab20', len(METRICS))
# https://scottplot.net/cookbook/4.1/colors/#category-20
colors = ["#1F77B4", "#AEC7E8", "#FF7F0E", "#FFBB78", "#2CA02C", "#98DF8A", "#D62728", 
    "#FF9896", "#9467BD", "#C5B0D5", "#8C564B", "#C49C94", "#E377C2", "#F7B6D2", 
    "#7F7F7F", "#C7C7C7", "#BCBD22", "#DBDB8D", "#17BECF", "#9EDAE5"]
metric_colors = {metric: colors[i] for i, metric in enumerate(METRICS)}

path = "/home/etienne/Dropbox/Projects/The Dutch Scaler Performance Indicator How much did my model actually learn/Results/"

# %% 

def gather_DS_results(M, P, rho):

    y_true = [1] * P + [0] * (M - P)

    results = []
    
    for metric in METRICS:
        
        # Dutch Draw Baseline
        DDB = DutchDraw.optimized_baseline_statistics(y_true, metric)['Max Expected Value'] 
        
        # Dutch Oracle
        DO = DutchScaler.upper_bound(y_true, metric, rho)
        
        bounds_dict = {"Metric":metric, "DDB": DDB, "DO": DO}
        
        for mu in np.linspace(DDB, DO):
            
            alpha, thetaopts = DutchScaler.optimized_indicator_inverted(y_true, metric, mu, rho)
            
            mu_alpha = DutchScaler.indicator_score(y_true, metric, alpha, thetaopts, rho)
            
            results_dict = {"Alpha": alpha,
                            "thetaopts":thetaopts,
                            "mu": mu,
                            "mu_alpha":mu_alpha
                            }
            
            combined_dict = {**bounds_dict, **results_dict}
            
            results.append(combined_dict)

    return pd.DataFrame(results)

# %%

def plot_and_save_1(df, M, P, rho, figure_name):
    
    plt.figure(figsize = (6,6))

    for metric, group, in df.groupby("Metric"):
        plt.plot(group["Alpha"], group["mu_alpha"], label = metric, color = metric_colors[metric])

    plt.xlabel(r'$\alpha$')
    plt.ylabel(r"$\mu_{\alpha}$")

    plt.ylim(0,1)
    
    plt.tight_layout()
    plt.legend()
    plt.savefig(path + figure_name + " M " + str(M) + " P " + str(P) + " rho " + str(rho) + ".png")
    plt.show()

def plot_and_save_2(df, M, P, rho, figure_name):
    
    plt.figure(figsize = (6,6))
    for metric, group, in df.groupby("Metric"):
        plt.plot(group["Alpha"], group["mu_alpha_scaled"], label = metric, color = metric_colors[metric])
    plt.ylabel(r'$\frac{\mu_\alpha - \mu_0}{\mu_1 - \mu_0}$')
    plt.xlabel(r'$\alpha$')
    
    plt.xlim(0,1)
    plt.ylim(0,1)
    
    plt.tight_layout()
    plt.legend()
    plt.savefig(path + figure_name + " M " + str(M) + " P " + str(P) + " rho " + str(rho) + ".png")

    plt.show()

# %% Figure 1-a

M_a = 100
P_a = 10
rho_a = 0.0
figure_name_a = "Figure-1a-"

df_a = gather_DS_results(M_a, P_a, rho_a)
plot_and_save_1(df_a, M_a, P_a, rho_a, figure_name_a)

# %% Figure 1-c

M_c = 100 
P_c = 10
rho_c = 0.05
figure_name_c = "Figure-1c-"

df_c = gather_DS_results(M_c, P_c, rho_c)
plot_and_save_1(df_c, M_c, P_c, rho_c, figure_name_c)


# %% Figure 1-b

def gather_scaled_DS_results(M, P, rho):

    y_true = [1] * P + [0] * (M - P)
    
    results = []
    
    for metric in METRICS:

        if metric == "G2":
            continue
        
        mu_0 = DutchScaler.lower_bound(y_true, metric)
        
        mu_1 = DutchScaler.upper_bound(y_true, metric, rho)

        bounds_dict = {"Metric":metric, "mu_0": mu_0, "mu_1": mu_1}
        
        for mu in np.linspace(mu_0, mu_1):
            
            alpha, thetaopts = DutchScaler.optimized_indicator_inverted(y_true, metric, mu, rho)
            
            mu_alpha = DutchScaler.indicator_score(y_true, metric, alpha, thetaopts, rho)
            
            mu_alpha_scaled = (mu_alpha - mu_0) / (mu_1 - mu_0)
            
            results_dict = {"Alpha": alpha,
                            "thetaopts":thetaopts,
                            "mu": mu,
                            "mu_alpha":mu_alpha,
                            "mu_alpha_scaled":mu_alpha_scaled
                            }
            
            combined_dict = {**bounds_dict, **results_dict}
            
            results.append(combined_dict)
    
    return pd.DataFrame(results)

# %% 

M_b = 100 
P_b = 10
rho_b = 0.05
figure_name_b = "Figure-1b"

df_b = gather_scaled_DS_results(M_b, P_b, rho_b)
plot_and_save_2(df_b, M_b, P_b, rho_b, figure_name_b)


# %%
axis_size = 14
def plot_and_save_3(df_a, df_b, df_c):

    fig, axs = plt.subplots(1, 3, figsize=(20, 6)) 
    
    for metric, group, in df_a.groupby("Metric"):
        axs[0].plot(group["Alpha"], group["mu_alpha"], label = metric, color = metric_colors[metric])
    axs[0].set_ylabel(r"$\mu_{\alpha}$", fontsize=axis_size)
    axs[0].set_xlabel(r'$\alpha$', fontsize=axis_size)
    
    axs[0].set_xlim(0,1)
    axs[0].set_ylim(0,1)

    for metric, group, in df_b.groupby("Metric"):
        axs[1].plot(group["Alpha"], group["mu_alpha_scaled"], color = metric_colors[metric])
    axs[1].set_ylabel(r'$\frac{\mu_\alpha - \mu_0}{\mu_1 - \mu_0}$', fontsize=axis_size)
    axs[1].set_xlabel(r'$\alpha$', fontsize=axis_size)
    
    axs[1].set_xlim(0,1)
    axs[1].set_ylim(0,1)


    for metric, group, in df_c.groupby("Metric"):
        axs[2].plot(group["Alpha"], group["mu_alpha"], color = metric_colors[metric])
    axs[2].set_ylabel(r"$\mu_{\alpha}$", fontsize=axis_size)
    axs[2].set_xlabel(r'$\alpha$', fontsize=axis_size)
    
    axs[2].set_xlim(0,1)
    axs[2].set_ylim(0,1)
    
    fig.legend(loc='upper center', ncol=len(METRICS), bbox_to_anchor=(0.51, 1), prop={'size': 14})
    
    fig.text(0.227, 0, r'(a) $\rho = 0$', ha='center', fontsize=16)
    fig.text(0.505, 0, r'(b) $\rho = 0$', ha='center', fontsize=16)
    fig.text(0.787, 0, r'(c) $\rho = 0.05$', ha='center', fontsize=16)
    
   # fig.tight_layout()
   
    plt.savefig(path + "Dutch-Scaler-Figure-1.png", bbox_inches="tight")
    
    plt.show()
    
# %%
plot_and_save_3(df_a, df_b, df_c)
