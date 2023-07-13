#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:00:17 2023

@author: etienne
"""

# TS and G2 are not linear in TP

# y_true = [1] * 1 + [0] * 13
# thetaopts = DutchDraw.optimized_baseline_statistics(y_true, "TS")['Argmax Expected Value']
# DD_expectation = DutchDraw.baseline_functions_given_theta(1, y_true, "TS")
# print("TS Value " + str(check_function(y_true, 0.0, [1], "TS")))
# Some code trying to show why alpha is below 0 for the G2.
# y_true = [1] * 2 + [0] * 28
# thetaopts = DutchDraw.optimized_baseline_statistics(y_true, "G2")['Argmax Expected Value']
# print(DutchDraw.optimized_baseline_statistics(y_true, "G2")['Max Expected Value'])
# print("Optimal Theta :" + str(thetaopts))
# print("G2 Value " + str(check_function(y_true, 0.0, thetaopts, "G2", beta = 1)))
# print(np.sqrt(thetaopts[0] * (1 - thetaopts[0])))

# y_true = [1] * 2 + [0] * 28
# DD_expectation = DutchDraw.baseline_functions_given_theta(0.6, y_true, "G2")
# print(DD_expectation)
# #Stel alpha = 0
# G2 = np.sqrt(0.6 * 0.4) #np.sqrt(TPR * TNR)
# print(G2)
