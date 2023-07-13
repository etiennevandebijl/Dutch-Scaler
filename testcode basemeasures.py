#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 20:28:10 2023

@author: etienne
"""

import random
import numpy as np

M = 100
P = 40
N = M - P

y_true = [1] * P + [0] * N
y_pred = [1] * P + [0] * N 
random.shuffle(y_pred)
TP = np.sum([a * b for a,b in zip(y_true, y_pred)])
FN = P - TP 
TN = np.sum([(1 - a) * (1 - b) for a,b in zip(y_true, y_pred)])
FP = N - TN

F1 = (2 * TP) / (2 * TP + FN + FP)
print(F1)
DDB = 2 * P / (P + M)
print(DDB)
if F1 > DDB:
    alpha_ = TN / N
    print(alpha_)
    alpha2 = 1 - ((P * 2 * (1 - F1)) / (N * F1))
    print(alpha2)
    
    true_P = 10
true_N = 20
r = [1] * (P - true_P) + [0] * (N - true_N)
random.shuffle(r)
y_pred = [1] * true_P + r + [0] * true_N