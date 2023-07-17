import numpy as np
import DutchDraw as DutchDraw

measure_dictionary = DutchDraw.measure_dictionary

def DSPI_Upperlimit(y_true, measure, rho, beta = 1):

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
        return P * (1 - rho) / ( rho * (N - P) + P)
    
    if measure in measure_dictionary['NPV']:
        return N * (1 - rho) / (rho * (P - N) + N)
    
    if measure in measure_dictionary['ACC']:
        return (1 - rho)
    
    if measure in measure_dictionary['BACC']:
        return (1 - rho)
    
    if measure in measure_dictionary['FBETA']:
        up = (1 + beta * beta) * P * (1 - rho)
        down = rho * (N - P) + P * (1 + beta * beta)
        return up/down
        
    if measure in measure_dictionary['MCC']:
        up = np.sqrt(P * N) * (1 - 2 * rho)
        down = np.sqrt( (rho * (N - P) + P)*( rho * (P - N) + N))
        return up/down
    
    if measure in measure_dictionary['J']:
        return 1 - 2 * rho
    
    if measure in measure_dictionary['MK']:
        up = N * P * (1 - 2 * rho)
        down = N * P - rho**2 * (P - N)**2 + rho * (P - N)**2 
        return up / down

    if measure in measure_dictionary['KAPPA']:
        up = 2 * P * N * (1 - 2 * rho)
        down = rho * (N - P)**2 + 2 * P * N
        return up/down
    
    if measure in measure_dictionary['FM']:
        up = np.sqrt(P) * (1 - rho)
        down = np.sqrt(P * (1 - rho) + N * rho)
        return up/down

    if measure in measure_dictionary['G2']:
        return 1 - rho
    
    if measure in measure_dictionary['TS']:
        up = P * (1 - rho)
        down = rho * N + P
        return up/down