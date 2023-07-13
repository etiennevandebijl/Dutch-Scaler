import numpy as np

M = 500
P = 23
theta = 0.2
score = 0.9
theta_star = round(theta * M) / M

def calculate_MK(a, t, m, p, n, mk):
    PPV_d = (a * p * (1 - t) + p * t)
    PPV_n = (a * (p - m * t) + m * t)
    PPV = PPV_d/PPV_n
    NPV_d = (a * n * t + n * (1 - t))
    NPV_n = (a * (m*t - p) + m * (1 - t))
    NPV = NPV_d / NPV_n
    return PPV + NPV - 1 - mk

def calculate_alpha(t, m, p, n, mk):
    a = (p - m * t) * (p - m * t) * mk
    b = (p - m * t) * (n - m * (mk + 1)) + m * p * (1 - t) + 2 * (p - m * t) * m * t * mk
    c = m * m * t * mk * (t - 1)
    
    a_1 = (-b + np.sqrt(b*b - 4 * a * c) ) / (2 * a)
    a_2 = (-b - np.sqrt(b*b - 4 * a * c) ) / (2 * a)
    return a_1, a_2

a_1, a_2 = calculate_alpha(theta_star, M, P, M - P, score)
print(a_1)
print(a_2)

result = calculate_MK(a_1, theta_star, M, P, M-P, score)
print(result)

results = []
for m in [10, 30, 50, 100, 150]:
    for p in range(1, m):
        for s in [0.1, 0.2, 0.4, 0.5, 0.6, 0.9]:
            for theta in [0.1, 0.2, 0.4, 0.5, 0.6, 0.9]:
                t_ = round(theta * m) / m
                if (p - m * t_) != 0:
                    a_1, a_2 = calculate_alpha(t_, m, p, m-p, s)
                    result = calculate_MK(a_1, t_, m, p, m-p, s)
                    results.append([m, p, s, t_, a_1, result])

import pandas as pd

df = pd.DataFrame(results, columns = ["M", "P", "MK","Theta*", "a_1","Result"])

# p - m* t mag geen 0 zijn