import DutchDraw as dutchdraw
import DutchScaler as dutchscaler
import random
import numpy as np

random.seed(123) # To ensure similar outputs

# Generate true and predicted labels

y_true = np.array(random.choices((0,1), k = 1000, weights = (0.9, 0.1)))
y_pred = y_true.copy()
flip_indices = np.random.choice(1000, size=50, replace=False)
y_pred[flip_indices] = 1 - y_pred[flip_indices]

######################################################
# Example function: bounds DutchScaler

print('Lower bound: {:06.4f}'.format(dutchscaler.lower_bound(y_true, measure= 'FBETA', beta = 2)))
print('Upper bound: {:06.4f}'.format(dutchscaler.upper_bound(y_true, measure= 'FBETA', beta = 2)))

print('')

######################################################
# Example function: find rho given score 0.95

print('$rho$: {:06.4f}'.format(dutchscaler.select_rho(y_true,  measure= 'FBETA', 0.95, beta = 2)))


######################################################
# Example function: acceptable max rho

print('Max rho: {:06.4f}'.format(dutchscaler.valid_rho_values(y_true,  measure= 'FBETA', beta = 2)))

######################################################
# Example function: find the DSPI
alpha, thetaopts = dutchscaler.optimized_indicator_inverted(y_true, measure = "FBETA", score = 0.9, rho = 0, beta = 2)
print('DSPI (alpha): {:06.4f}'.format(alpha))


