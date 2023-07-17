import numpy as np
import DutchDraw as DutchDraw

measure_dictionary = DutchDraw.measure_dictionary

def DSPI_v2(y_true, measure, alpha, rho = 0, beta = 1):
    '''
    
    This function derives the DSPI score without the need to find the optimal 
    thetaopts.

    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    measure : TYPE
        DESCRIPTION.
    rho : TYPE, optional
        DESCRIPTION. The default is 0.
    beta : TYPE, optional
        DESCRIPTION. The default is 1.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    measure = measure.upper()

    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)
    
    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "KAPPA", "FM", "TS"]:
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("No direct function is derived yet")
 
    if measure in measure_dictionary['PPV']:
        up = alpha * P * (M - 1 - M * rho) + P
        down = alpha * M * (rho * (N - P) + P - 1) + M
        return up/down

    if measure in measure_dictionary['NPV']:
        up = alpha * N * (M - 1 - M * rho) + N
        down = alpha * M * (rho * (P - N) + N - 1) + M
        return up/down

    if measure in measure_dictionary['FBETA']:
        up = (1 + beta * beta) * P * (1 - alpha * rho)
        down = alpha * (rho * (N - P) -N) + M + P * beta * beta
        return up / down

    if measure in measure_dictionary['ACC']:
        return  (alpha * (min(P , N) - M * rho) + max(P, N)) * (1 / M)

    if measure in measure_dictionary['FM']:
        up = np.sqrt(P) * (1 - alpha * rho)
        down = np.sqrt(alpha * (rho * (N - P) -N) + M)
        return up / down

    if measure in measure_dictionary['J']:
        return alpha * (1 - 2 * rho)

    if measure in measure_dictionary['BACC']:
        return alpha * (1 - 2 * rho) * 0.5  + 0.5

    if measure in measure_dictionary['KAPPA']:
        up = 2 * P * N * alpha * (1 - 2 * rho)
        down = M * min(P,N) + alpha * ( rho * (N - P)**2 - (min(N,P)**2) + P * N)
        return up / down
    
    if measure in measure_dictionary['TS']:
        if (P > 1) or (alpha * N * (1 - 2 * rho) <= 1):
            up = P * (1 - alpha * rho)
            down = M - N * alpha * (1 - rho)
            return up/down
        else:
            up = alpha * (N - M * rho) + 1
            down = M + N - alpha * N * (1 - M * rho)
    
    '''
    Open tickets: Derive optimal thetaopts for G2, MCC, MK
    '''
    


















