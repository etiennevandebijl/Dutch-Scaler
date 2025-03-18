# The Dutch Scaler Performance Indicator package

This repository contains the code for the article "The Dutch Scaler performance indicator: How much did my model actually learn?".

## Environment  

- The code is designed to run in an **Anaconda** environment.  
- Ensure all dependencies are installed before running the scripts.  

## Getting Started  

1. Clone this repository:  
   ```bash
   git clone https://github.com/etiennevandebijl/Dutch-Scaler.git  
   cd Dutch-Scaler  

2. Set up your Anaconda environment (if not already configured):
    ```bash
    conda env create -f environment-DutchScaler.yml   
    conda activate DutchScaler  
    ```
3. Use the DutchScaler by importing in python:

```python
import DutchScaler as dutchscaler
```

## Usage

As example, we first generate the true and predicted labels.

```python
import DutchScaler as dutchscaler
import random
import numpy as np

random.seed(123) # To ensure similar outputs

# Generate true and predicted labels

y_true = np.array(random.choices((0,1), k = 1000, weights = (0.9, 0.1)))
y_pred = y_true.copy()
flip_indices = np.random.choice(1000, size=50, replace=False)
y_pred[flip_indices] = 1 - y_pred[flip_indices]

```

To get the boundaries of the Dutch Scaler Performance Indicator score:
```python
print('Lower bound: {:06.4f}'.format(dutchscaler.lower_bound(y_true, measure= 'FBETA', beta = 2)))
print('Upper bound: {:06.4f}'.format(dutchscaler.upper_bound(y_true, measure= 'FBETA', beta = 2)))
```

To find the DSPI score for a given score: 
```python
alpha, thetaopts = dutchscaler.optimized_indicator_inverted(y_true, measure = "FBETA", score = 0.9, rho = 0, beta = 2)
print('DSPI (alpha): {:06.4f}'.format(alpha))
```

## Citation
If you use this code in your research, please cite:

    E.P. van de Bijl, J.G. Klein, J. Pries, S. Bhulai, and R.D. van der Mei, ``The Dutch Scaler performance indicator: How much did my model actually learn?'', 2025. 

## License

[MIT](https://choosealicense.com/licenses/mit/)
