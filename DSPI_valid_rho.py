import numpy as np
import DutchDraw as DutchDraw

measure_dictionary = DutchDraw.measure_dictionary

def DSPI_valid_rho(y_true, measure, beta = 1):
    
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
        return 0.5
    
    if measure in measure_dictionary['NPV']:
        return 0.5
    
    if measure in measure_dictionary['ACC']:
        return min(N,P)/M

    if measure in measure_dictionary['BACC']:
        return 0.5
    
    if measure in measure_dictionary['FBETA']:
        return N/(2*N+P*beta*beta)
        
    if measure in measure_dictionary['MCC']:
        return 0.5
    
    if measure in measure_dictionary['J']:
        return 0.5
    
    if measure in measure_dictionary['MK']:
        return 0.5
    
    if measure in measure_dictionary['KAPPA']:
        return 0.5
    
    if measure in measure_dictionary['FM']:
        return N / (3 * N + P)

    if measure in measure_dictionary['G2']:
        return 0.5
    
    if measure in measure_dictionary['TS']:
        '''
        Open ticket in the article to fix this, we have some solution.
        '''
        if P > 1:
            return N / (N + M)
        else:
            return (N) / (N + M)
        