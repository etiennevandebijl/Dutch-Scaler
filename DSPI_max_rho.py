
import numpy as np
import DutchDraw as DutchDraw

measure_dictionary = DutchDraw.measure_dictionary

def DSPI_max_rho(y_true, measure, score, beta = 1):    
    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "G2", "TS"]:
        if measure == "MCC":
            raise ValueError("The DSPI is not YET supported for this measure")
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("The DSPI is not supported for this measure")
        
    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)
    
    if measure in measure_dictionary['ACC']:
        return (1 - score)

    if measure in measure_dictionary['BACC']:
        return (1 - score)
    
    if measure in measure_dictionary['J']:
        return 0.5 - 0.5 * score

    if measure in measure_dictionary['G2']:
        return 1 - score

    if measure in measure_dictionary['PPV']:
        return P * (1 - score) / ( score * (N - P) + P)
    
    if measure in measure_dictionary['NPV']:
        return N * (1 - score) / (score * (P - N) + N)
    
    if measure in measure_dictionary['FBETA']:
        up = (1 + beta * beta) * P * (1 - score)
        down = score * (N - P) + P * (1 + beta * beta)
        return up/down
        
    if measure in measure_dictionary['TS']:
        up = P * (1 - score)
        down = score * N + P
        return up/down
    
    if measure in measure_dictionary['KAPPA']:
        up = 2 * P * N * (1 - score)
        down = score * (N - P)**2 + 4 * P * N
        return up/down    
    
    
    if measure in measure_dictionary['FM']:
        term = (P - N) / P
        up = term * score * score + score * np.sqrt(term * term * score * score + 4 * (N / P))
        return 1 - up/2

    if measure in measure_dictionary['MK']:
        D = score * score * (P - N)**4 + 4 * N * P * score**2 * (P-N)**2 + 4 * N**2 * P**2
        up = 2 * N * P - np.sqrt(D)
        down = 2 * score * (P - N)**2
        return 0.5 + up / down
    
    if measure in measure_dictionary['MCC']:
        '''
        Open ticket: TO DO
        '''
        return 0
    

    


    

    
