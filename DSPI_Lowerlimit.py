import numpy as np
import DutchDraw as DutchDraw

measure_dictionary = DutchDraw.measure_dictionary

def DSPI_Lowerlimit(y_true, measure, beta = 1):

    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "G2", "TS"]:
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("The DSPI is not supported for this measure")
        
    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)
    
    
    if measure in measure_dictionary['PPV']:
        return P / M
    
    if measure in measure_dictionary['NPV']:
        return N / M
    
    if measure in measure_dictionary['ACC']:
        return max(P,N)/M
    
    if measure in measure_dictionary['BACC']:
        return 0.5
    
    if measure in measure_dictionary['FBETA']:
        return (1 + beta * beta) * P / (M + P * beta * beta)
        
    if measure in measure_dictionary['MCC']:
        return 0
    
    if measure in measure_dictionary['J']:
        return 0
    
    if measure in measure_dictionary['MK']:
        return 0

    if measure in measure_dictionary['KAPPA']:
        return 0
    
    if measure in measure_dictionary['FM']:
        return np.sqrt(P/M)

    if measure in measure_dictionary['TS']:
        return P / M