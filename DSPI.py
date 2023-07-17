import numpy as np
import DutchDraw as DutchDraw

measure_dictionary = DutchDraw.measure_dictionary

def DSPI(y_true, measure, alpha, thetaopts, rho = 0, beta = 1):
    
    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "G2", "TS"]:
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("The DSPI is not supported for this measure")
        
    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)
    
    t = thetaopts[0]
    TP = alpha * P * (1 - rho - t) + P * t 
    TN = alpha * N * (t - rho) + N * (1 - t) 
    FN = alpha * P * (rho - 1 + t) + P * (1 - t)
    FP = alpha * N * (rho - t) + N * t

    if measure in measure_dictionary['PPV']:
        return TP / (TP + FP)

    if measure in measure_dictionary['NPV']:
        return TN / (TN + FN)

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
        return (TP * TN - FP * FN)/(np.sqrt((TP + FP) * (TN + FN) * P * N))

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
        return np.sqrt(TPR * PPV)

    if measure in measure_dictionary['G2']:
        TPR = TP / P
        TNR = TN / N
        return np.sqrt(TPR * TNR)

    if measure in measure_dictionary['TS']:
        return TP / (TP + FN + FP)