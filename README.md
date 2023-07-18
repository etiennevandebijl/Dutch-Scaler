# The DutchScaler package

DutchDraw is a Python package for constructing baselines in binary classification.

## Paper

This package is an implementation of the ideas from `The Dutch Scaler Performance Indicator: What did my model actually learn?'.

### Citation

If you have used the DutchDraw package, please also cite: (Coming soon)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the package

## Method

## Reasons to use

### List of all included measures

| Measure                                                                  |                                                                                                     Definition                                                                                                     |
| ------------------------------------------------------------------------ | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| PPV                                                                      |                                                                                                   TP / (TP + FP)                                                                                                   |
| NPV                                                                      |                                                                                                   TN / (TN + FN)                                                                                                   |
| ACC, ACCURACY                                                            |                                                                                                    (TP + TN) / M                                                                                                    |
| BACC, BALANCED ACCURACY                                                  |                                                                                                   (TPR + TNR) / 2                                                                                                   |
| FBETA, FSCORE, F, F BETA, F BETA SCORE, FBETA SCORE                      |                                                    ((1 + β`<sup>`2`</sup>`) * TP) / ((1 + β`<sup>`2`</sup>`) * TP + β`<sup>`2`</sup>` * FN + FP)                                                    |
| MCC, MATTHEW, MATTHEWS CORRELATION COEFFICIENT                           |                                                                             (TP * TN - FP * FN) / (sqrt((TP + FP) * (TN + FN) * P * N))                                                                             |
| BM, BOOKMAKER INFORMEDNESS, INFORMEDNESS                                 |                                                                                                    TPR + TNR - 1                                                                                                    |
| MK                                                                       |                                                                                                    PPV + NPV - 1                                                                                                    |
| COHEN, COHENS KAPPA, KAPPA                                               | (P`<sub>`o`</sub>` - P`<sub>`e`</sub>`) / (1 - P`<sub>`e`</sub>`) with P`<sub>`o`</sub>` = (TP + TN) / M and `<br>` P`<sub>`e`</sub>` = ((TP + FP) / M) * (P / M) + ((TN + FN) / M) * (N / M) |
| G1, GMEAN1, G MEAN 1, FOWLKES-MALLOWS, FOWLKES MALLOWS, FOWLKES, MALLOWS |                                                                                                   sqrt(TPR * PPV)                                                                                                   |
| G2, GMEAN2, G MEAN 2                                                     |                                                                                                   sqrt(TPR * TNR)                                                                                                   |
| TS, THREAT SCORE, CRITICAL SUCCES INDEX, CSI                             |                                                                                                 TP / (TP + FN + FP)                                                                                                 |


## License

[MIT](https://choosealicense.com/licenses/mit/)