"""
@author : HERTZOG Thibaut & ROGUET William
"""

# Importation des librairies nécessaires

import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import scipy.special as sp
import scipy.optimize as opt
import sklearn
from sklearn import preprocessing, metrics, datasets, model_selection, utils

# Définition des constantes
X = sklearn.datasets.load_digits().data
Y = sklearn.datasets.load_digits().target
N = len(X)
d = len(X[0])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)

x_train_ones = np.column_stack((np.ones(len(x_train)), x_train))
x_test_ones = np.column_stack((np.ones(len(x_test)), x_test))

y_train = np.asarray(y_train, dtype=float)
y_test = np.asarray(y_test, dtype=float)

number_of_classes = len(np.unique(y_train))