import numpy as np
import DutchDraw as DutchDraw
from DSPI_valid_rho import DSPI_valid_rho

measure_dictionary = DutchDraw.measure_dictionary

def DSPI_inverse(y_true, measure, score, rho = 0, beta = 1):
    '''
    This function derives alpha and theta when knowing the realized performance 
    metric score.

    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    measure : TYPE
        DESCRIPTION.
    score : TYPE
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
    TYPE
        DESCRIPTION.

    '''
    measure = measure.upper()

    P = np.int64(sum(y_true))
    M = np.int64(len(y_true))
    N = np.int64(M - P)

    # Check if DSPI is supported for the corresponding metric
    check_measure = False
    for m in ["PPV", "NPV", "ACC", "BACC", "FBETA", "MCC", "J", "MK", "KAPPA", "FM", "G2", "TS"]:
        if measure in measure_dictionary[m]:
            check_measure = True
    if not check_measure:
        raise ValueError("The DSPI is not supported for this measure")

    # Check if the realized score outperforms the DDB
    baseline = DutchDraw.optimized_baseline_statistics(y_true, measure)
    if baseline['Max Expected Value'] == score:
        if measure != "G2":
            if measure != "KAPPA":
                return 0, baseline['Argmax Expected Value']
    if baseline['Max Expected Value'] > score:
        raise ValueError("De score must outperform the Dutch Draw.")

    valid_rho = DSPI_valid_rho(y_true, measure, beta)
    if rho > valid_rho:
        raise ValueError("De score outperforms the valid rho. Alpha would be bigger than 1")
        
    if measure in measure_dictionary['J']:
        alpha = score / (1 - 2 * rho)
        thetaopts = [i/M for i in range(0, M + 1)]
        return  alpha, thetaopts

    if measure in measure_dictionary['BACC']:
        alpha = (2 * score - 1) / (1 - 2 * rho)
        thetaopts = [i/M for i in range(0, M + 1)]
        return alpha, thetaopts

    if measure in measure_dictionary['PPV']:
        alpha = (M * score - P) / (M * score - P + P * M * (1 - score) + rho * M * (score * (P - N) - P))
        thetaopts = [1.0 / M]
        return alpha, thetaopts

    if measure in measure_dictionary['NPV']:
        alpha = ((N - score * M) ) / ((score * M * (N - 1) + N * (1 - M) + rho * M * (score * (P - N) + N)))
        thetaopts = [(M - 1) / M]
        return alpha, thetaopts

    if measure in measure_dictionary['FBETA']:
        alpha = (P * (1 + beta*beta) * (1 - score) - score * N) / (score * ( rho * (N - P) - N) + (1 + beta*beta) * P * rho)
        thetaopts = [1]
        return alpha, thetaopts

    if measure in measure_dictionary['ACC']:
        alpha = (M * score - max(P, N))  / (min(P, N) - M * rho)
        if P < N:
            thetaopts = [0]
        elif P > N:
            thetaopts = [1]
        else:
            thetaopts =  [i/M for i in range(0, M+1)]
        return alpha, thetaopts

    if measure in measure_dictionary['KAPPA']:
        if (P == M or P == 0):
            '''
            Open ticket: Should this not lead to alpha = 0? Below result is from article. Needs check.
            '''
            return 1, [ ( M - 1 ) / M ] 
        else:
            alpha = (score * M * min(P,N)) / (2 * N * P * (1 - 2 * rho) + score * ((min(P,N)**2) - N * P - rho * (P - N)**2)) 
            if P == N:
                thetaopts = [i/M for i in range(0, M + 1)]
            if P > N: 
                thetaopts = [1]
            if N > P:
                thetaopts = [0]   
            return alpha, thetaopts

    if measure in measure_dictionary['FM']:
        thetaopts = [1]
        if rho == 0:
            alpha = (M / N) - (P / (score * score * N))
        else:
            a = rho * rho * P
            b = - 2 * rho * P - score * score * (rho * (N - P) - N)
            c = P - M * score * score 
            '''
            Open ticket: Which solution of the following equations is valid?
            '''
            a_1 = (-b + np.sqrt(b * b - 4 * a * c) ) / (2 * a)
            a_2 = (-b - np.sqrt(b * b - 4 * a * c) ) / (2 * a)
            
            answer = False
            if a_1 >= 0 and a_1 <= 1:
                alpha = a_1
                answer = True
            if a_2 >= 0 and a_2 <= 1:
                alpha = a_1
                answer = True
            if answer == False:
                raise ValueError("FM PROBLEM")
                
        return alpha, thetaopts

    if measure in measure_dictionary['MK']:
        alpha = np.inf
        thetaopts = []
        for t in baseline['Argmax Expected Value']:

            term = rho * (N - P) + P - M * t
            if np.abs(term)<0.0000001:
                a_1 = (M * M * t * (1 - t) * score) / ((1 - 2 * rho) * P * N)
            else:
                a = term * term * score
                b = -1 * term * M * score * (1 - 2 * t) + N * P * (1 - 2 * rho)
                c = -1 * M * M * t * score * (1 - t)
                a_1 = (-b + np.sqrt(b * b - 4 * a * c) ) / (2 * a) #Checked 
            if a_1 > 1.00000002: 
                raise ValueError("Alpha bigger than 1 MK")
                
            if a_1 == alpha:
                thetaopts.append(t)
            if a_1 < alpha:
                alpha = a_1
                thetaopts = [t]
        return alpha, thetaopts

    if measure in measure_dictionary['TS']:
        if P > 1:
            alpha = (M * score - P) / (N * score * (1 - rho) - P * rho) 
            thetaopts = [1]
        else:
            if N > (1 / score):
                alpha = (score * (M + N) - 1) / (score * N - M * score * N * rho - M * rho + N)
                thetaopts = [1 / M]
            elif score == 1 / N:
                alpha = score * (1 / (1 - 2 * rho))
                thetaopts = [i/M for i in range(1, M + 1)]
            else:
                alpha = (M * score - P) / (N * score * (1 - rho) - P * rho) 
                thetaopts = [1]
        return alpha, thetaopts


    if measure in measure_dictionary['G2']:
        # As G2 is not linear in TP, it can happen that alpha is negative. 
        alpha = np.inf
        thetaopts = []
        for t in baseline['Argmax Expected Value']:
            if (1 - rho - t) * (t - rho) == 0:
                a_1 = (score * score  - t *(1 - t)) / (1 - rho - 2 * t + 2 * t* t)
            else:
                a = (1 - rho - t) * (t - rho)
                b = ((2 * t * t) - (2 * t) + 1 - rho)
                c =  t - ( (t * t) + (score * score))
                ''' 
                Open ticket: Proof dat a_1 is always in [0,1] and a_2 is not
                '''
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

    if measure in measure_dictionary['MCC']:
        alpha = np.inf
        thetaopts = []
        for t in baseline['Argmax Expected Value']:
            a = score * score * (P - M * t + rho * (N - P) ) * (P - M * t + rho * (N - P)) + (P * N) * (1 - 2 * rho)**2 
            b = -1 * score * score * M * (P - M * t + rho * (N - P)) * (1 - 2 * t)
            c = - M * M * t * (1 - t) * score * score    
            '''
            Open ticket: In article, it needs to be shown that this is smaller than 1.
            '''
            a_1 = (-b + np.sqrt(b * b - 4 * a * c) ) / (2 * a) #Checked in article
            if a_1 > 1.00000002: 
                print(a_1)
                raise ValueError("Alpha bigger than 1 MCC")
            if a_1 == alpha:
                thetaopts.append(t)
            if a_1 < alpha:
                alpha = a_1
                thetaopts = [t]
        return alpha, thetaopts
    




    



