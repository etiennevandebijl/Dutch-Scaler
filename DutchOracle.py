import numpy as np
import DutchDraw as DutchDraw
import pandas as pd
import matplotlib.pyplot as plt
import math

measure_dictionary = DutchDraw.measure_dictionary

def dutch_oracle_determine_alpha(y_true, measure, score, beta=1):
    measure = measure.upper()

    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)

    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "G2", "TS"]:
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("Alpha cannot be calculated for the given measure.")

    baseline = DutchDraw.optimized_baseline_statistics(y_true, measure)
    if baseline['Max Expected Value'] == score:
        if measure != "G2":
            if measure != "KAPPA":
                return 0, baseline['Argmax Expected Value']

    if baseline['Max Expected Value'] > score:
        raise ValueError("De score must outperform the Dutch Draw.")

    if measure in measure_dictionary['PPV']:
        alpha = (M * score - P) / (M * score - P + P * M * (1 - score))
        thetaopts = [1.0 / M]
        return alpha, thetaopts

    if measure in measure_dictionary['NPV']:
        alpha = ((N - score * M) ) / ((score * M * (N - 1) + N * (1 - M)))
        thetaopts = [(M - 1) / M]
        return alpha, thetaopts

    if measure in measure_dictionary['ACC']:
        alpha = (M * score - max(P, N))  / min(P, N)
        if P < N:
            thetaopts = [0]
        elif P > N:
            thetaopts = [1]
        else:
            thetaopts =  [i/M for i in range(0, M+1)]
        return alpha, thetaopts

    if measure in measure_dictionary['BACC']:
        alpha = 2*score - 1
        thetaopts = [i/M for i in range(0, M+1)]
        return alpha, thetaopts

    if measure in measure_dictionary['FBETA']:
        alpha = 1 - (( P * (1 + beta*beta) * (1 - score)) / (N * score) )
        thetaopts = [1]
        return alpha, thetaopts

    if measure in measure_dictionary['J']:
        alpha = score
        thetaopts = [i/M for i in range(0, M + 1)]
        return  alpha, thetaopts

    if measure in measure_dictionary['FM']:
        alpha = (M * score * score - P) / (N * score * score)
        thetaopts = [1]
        return alpha, thetaopts

    if measure in measure_dictionary['KAPPA']:
        if (P == M or P == 0):
            return 1, [ (M-1) / M]
        else:
            alpha = (score * M) / (2 * max(N,P) + score * (min(N,P) - max(N,P))) 
            if P == N:
                thetaopts = [i/M for i in range(0, M+1)]
            if P > N:
                thetaopts = [1]
            if N > P:
                thetaopts = [0]   
            return alpha, thetaopts

    if measure in measure_dictionary['MK']:
        alpha = np.inf
        thetaopts = []
        for t in baseline['Argmax Expected Value']:
            if P - M * t == 0:
                continue
            a = (P - M * t) * (P - M * t) * score
            b = N * P + (P - M * t) * M * score * (2 * t - 1)
            c = M * M * t * score * (t - 1)
            
            #print(- b / (2*a))
            a_1 = (-b + np.sqrt(b * b - 4 * a * c) ) / (2 * a)
            a_2 = (-b - np.sqrt(b * b - 4 * a * c) ) / (2 * a) #Should not give satis results, needs check
            print(a_1)
            print(a_2)
            if a_1 < 0:
                raise ValueError("Alpha not positive MK")
            if a_1 > 1.00000002: 
                raise ValueError("Alpha bigger than 1 MK")
                
            if a_1 == alpha:
                thetaopts.append(t)
            if a_1 < alpha:
                alpha = a_1
                thetaopts = [t]
        return alpha, thetaopts

    if measure in measure_dictionary['MCC']:
        alpha = np.inf
        thetaopts = []
        for t in baseline['Argmax Expected Value']:
            a = (P - M * t) * (P - M * t) + (P * N) / (score * score)
            b = -1 * M * (P - M * t) * (1 - 2 * t)
            c = - M * M * t * (1 - t)
            
            a_1 = (-b + np.sqrt(b * b - 4 * a * c) ) / (2 * a)
#            a_2 = (-b - np.sqrt(b*b - 4 * a * c) ) / (2 * a) #Should not give satis results, needs check

            if a_1 < 0:
                raise ValueError("Alpha not positive MCC")
            if a_1 > 1.00000002: 
                raise ValueError("Alpha bigger than 1 MCC")
            if a_1 == alpha:
                thetaopts.append(t)
            if a_1 < alpha:
                alpha = a_1
                thetaopts = [t]
        return alpha, thetaopts
    
    if measure in measure_dictionary['G2']:
        # As G2 is not linear in TP, it can happen that alpha is negative. 
        alpha = np.inf
        thetaopts = []
        for t in baseline['Argmax Expected Value']:
            if (1 - t) * t == 0:
                print("problem with t G2")
                continue
            a = (1 - t) * t
            b = ((2 * t * t) - (2 * t) + 1)
            c =  t - ( (t * t) + (score * score))
            
            a_1 = (-b + np.sqrt(b * b - 4 * a * c) ) / (2 * a) 
            # a_2 = (-b - np.sqrt(b * b - 4 * a * c) ) / (2 * a) 
            if a_1 > 1: 
                raise ValueError("Alpha bigger than 1 G2")
                
            if a_1 == alpha:
                thetaopts.append(t)
            if a_1 < alpha:
                alpha = a_1
                thetaopts = [t]
        return alpha, thetaopts

    if measure in measure_dictionary['TS']:
        if P > 1:
            alpha = (M * score - P) / (N * score) 
            thetaopts = [1]
        else:
            if N > (1 / score):
                alpha = 1 - ((N + 1) * (1 - score) / (N * (1 + score)))
                thetaopts = [1 / M]

            elif score == 1 / N:
                alpha = score
                thetaopts = [i/M for i in range(1, M + 1)]
            else:
                alpha = 1 - ((P * (1 - score)) / (N * score)) 
                thetaopts = [1]
        return alpha, thetaopts
    
def check_function(y_true, alpha, thetaopts, measure, beta = 1):
    
    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)
    
    t = thetaopts[0]
    TP = alpha * P * (1 - t) + P * t 
    TN = alpha * N * t + N * (1 - t) 
    FN = - alpha * P * (1 - t) + P * (1 - t)
    FP = - alpha * N * t + N * t


    if measure in measure_dictionary['TP']:
        return TP

    if measure in measure_dictionary['TN']:
        return TN

    if measure in measure_dictionary['FP']:
        return FP

    if measure in measure_dictionary['FN']:
        return FN

    if measure in measure_dictionary['TPR']:
        return TP / P

    if measure in measure_dictionary['TNR']:
        return TN / N

    if measure in measure_dictionary['FPR']:
        return FP / N

    if measure in measure_dictionary['FNR']:
        return FN / P

    if measure in measure_dictionary['PPV']:
        return TP / (TP + FP)

    if measure in measure_dictionary['NPV']:
        return TN / (TN + FN)

    if measure in measure_dictionary['FDR']:
        return FP / (TP + FP)

    if measure in measure_dictionary['FOR']:
        return FN / (TN + FN)

    if measure in measure_dictionary['ACC']:
        return (TP + TN) / M

    if measure in measure_dictionary['BACC']:
        TPR = TP / P
        TNR = TN / N
        return (TPR + TNR) / 2

    if measure in measure_dictionary['FBETA']:
        beta_squared = beta ** 2
        return (1 + beta_squared) * TP / (((1 + beta_squared) * TP) + (beta_squared * FN) + FP)

    if measure in measure_dictionary['MCC']:
        return (TP * TN - FP * FN)/(math.sqrt((TP + FP) * (TN + FN) * P * N))

    if measure in measure_dictionary['J']:
        TPR = TP / P
        TNR = TN / N
        return TPR + TNR - 1

    if measure in measure_dictionary['MK']:
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        return PPV + NPV - 1

    if measure in measure_dictionary['KAPPA']:
        P_o = (TP + TN) / M
        P_yes = ((TP + FP) / M) * (P / M)
        P_no = ((TN + FN) / M) * (N / M)
        P_e = P_yes + P_no
        return (P_o - P_e) / (1 - P_e)

    if measure in measure_dictionary['FM']:
        TPR = TP / P
        PPV = TP / (TP + FP)
        return math.sqrt(TPR * PPV)

    if measure in measure_dictionary['G2']:
        TPR = TP / P
        TNR = TN / N
        return np.sqrt(TPR * TNR)

    if measure in measure_dictionary['TS']:
        return TP / (TP + FN + FP)

metrics = ["ACC","BACC", "FBETA", "FM", "G2" ,"NPV", "MCC", "KAPPA", "PPV", "TS", "MK", "J"]
#metrics = ["J", "KAPPA", "MCC", "MK"]
metrics = ["MK"]
results = []

for m in [40]:
    for p in [9]:
        y_true = [1] * p + [0] * (m - p)
        for metric in metrics:
            baseline = DutchDraw.optimized_baseline_statistics(y_true, metric)['Max Expected Value']       
            for s in np.linspace(baseline, 1): 
                alpha, thetaopts = dutch_oracle_determine_alpha(y_true, metric, s)
                
                reverse_score = check_function(y_true, alpha, thetaopts, metric, beta = 1)
                results.append([m, p, s, metric, alpha, thetaopts, np.abs(s - reverse_score), baseline])

df = pd.DataFrame(results, columns = ["M", "P", "Score DO", "Metric", "Alpha", "Thetaopts", "Reverse_score", "Baseline"])

# def derivative_fbeta(M, P, alpha, beta):
#     N = M - P
#     d = N * (1 + beta * beta) * P
#     n = (alpha * -N + M + P * beta**2)**2
#     return d /n

# def fbeta(M, P, alpha, beta):
#     N = M - P
#     d = (1 + beta * beta) * P
#     n = alpha * -N + M + P * beta * beta
#     return d/n

# alpha_ex = 0.75
# der = derivative_fbeta(m, p, alpha_ex, 1)
# FBaR =  fbeta(m, p, alpha_ex, 1)
# b = FBaR - der *alpha_ex 

# y_1 = der * (alpha_ex - 0.2) + b
# y_2 = der * (alpha_ex + 0.2) + b


# x = np.linspace(0, 1)
# y = [derivative_fbeta(m, p, i, 1) for i in x]
# y_bla = [1 for i in x]
# plt.figure(figsize = (10,10))
# plt.plot(x, y)e)
# plt.plot(x, y_bla)
# plt.xlabel("Alpha")
# plt.show()

plt.figure(figsize = (8,8))
for metric, group in df.groupby("Metric"):
    #plt.plot(group["Alpha"], group["Score DO"], label = metric)
    plt.plot(group["Score DO"], group["Alpha"], label = metric)
 #   plt.plot([alpha_ex - 0.2, alpha_ex + 0.2], [y_1, y_2], label = "Derivative")
plt.legend()
# plt.ylabel(r'$\mu_\alpha$')
plt.xlabel(r"$\overline{\mu}$")
# plt.xlabel(r'$\alpha$')
plt.ylabel(r"DSPI")
plt.ylim(0,1)
plt.show()

plt.figure(figsize = (8,8))
for metric, group in df.groupby("Metric"):
    plt.plot(group["Alpha"], group["Score DO"], label = metric)
plt.legend()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$\mu_{\alpha}$")
plt.xlim(0,1)

plt.show()

plt.figure(figsize = (8,8))
for metric, group in df.groupby("Metric"):
    group["Scaled-Score"] = (group["Score DO"] - group["Baseline"].mean()) / (1 - group["Baseline"].mean())
    plt.plot(group["Alpha"], group["Scaled-Score"], label = metric)

plt.legend()
plt.ylabel(r'$\frac{\mu_\alpha - \mu_0}{\mu_1 - \mu_0}$')
plt.xlabel(r'$\alpha$')
#plt.title("Minimal Alpha required to achieve realised score M = 30, P = 10")
plt.xlim(0,1)
plt.show()



# plt.figure(figsize = (10,10))
# for metric, group in df.groupby("Metric"):
#     group["Scaled-Score"] = np.sqrt(((group["Score"] - group["Baseline"].mean()) / (1 - group["Baseline"].mean()) - group["Alpha"])**2)
#     plt.plot(group["Alpha"], group["Scaled-Score"], label = metric)
# plt.legend()
# plt.ylabel("|(Score - DDB) / (1 - DDB) - alpha|")
# plt.xlabel("Alpha")
# plt.title("Minimal Alpha required to achieve realised score M = 30, P = 4")
# plt.show()



# =============================================================================
# Hier een bewijs dat niet altijd alle alpha's hetzelfde zijn 
# =============================================================================

# metrics = ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "TS", "G2"]

# for m in [30]:
#     for p in [5]:
#         y_true = [1] * p + [0] * (m - p)
#         random_flip = np.random.rand(m)
#         y_pred = [1 - x if z < 0.05 else x for x, z in zip(y_true, random_flip)]
        
#         tp = DutchDraw.measure_score(y_true, y_pred, "TP")
#         tn = DutchDraw.measure_score(y_true, y_pred, "TN")
#         fp = DutchDraw.measure_score(y_true, y_pred, "FP")
#         fn = DutchDraw.measure_score(y_true, y_pred, "FN")
#         results = {"tp": [tp, "*", p, 0, 1], "tn": [tn, "*", 0, (m - p), 0]}
#         results["fn"] = [fn, "*", 0, 0, 1]
#         results["fp"] = [fp, "*", 0, 0, 0]
#         for metric in metrics:
#             sm = DutchDraw.measure_score(y_true, y_pred, metric)
#             alpha, thetaopts = dutch_oracle_determine_alpha(y_true, metric, sm)
#             TPa = alpha * p + (1 - alpha) * p * thetaopts[0]
#             TNa = alpha * (m - p) * thetaopts[0] + (1 - thetaopts[0]) * (m - p)
#             results[metric] = [sm, alpha, TPa, TNa, thetaopts[0]]
            
# df = pd.DataFrame(results).T
# df.columns = ["Score", "Alpha", "TPa", "TNa", "Thetaopt"]
# print(df)



