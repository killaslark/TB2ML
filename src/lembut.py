"""
NN from scratch.
"""

# Authors: Gilang Ardyamandala Al Assyifa <gilangardya@gmail.com>
#          Bobby Indra Nainggolan <kodok.botak12@gmail.com>
#          Mico <>

import numpy as np

def identity(x):
	return x

def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def relu(x):
	return max(0, x)

def binary_step(x):
	if x < 0:
		return 0
	return 1

def soft_sign(x):
	return x / (1 + np.abs(x))

def gaussian(x):
	return np.exp(-x)