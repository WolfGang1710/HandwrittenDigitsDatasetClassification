import numpy as np


def softmax(x):
    """
    Calcule la fonction softmax pour un vecteur x.
    :param x: vecteur numpy
    :return: vecteur numpy
    """
    numerateur = np.exp(x - np.max(x, axis=1, keepdims=True))  # Pour éviter les overflow / stabilité numérique
    return numerateur / np.sum(numerateur, axis=1, keepdims=True)

def transform_labels(y):
    """
    Encode les labels en one-hot.
    Exemple : on a une liste de 6 fleures chacune peut avoir une des 3 classes
    Entrée: [
           1,
           3,
           3,
           2,
           1,
           2
          ]

  Sortie: [
           [1,0,0], # class 1
           [0,0,1], # class 3
           [0,0,1], # class 3
           [0,1,0], # class 2
           [1,0,0], # class 1
           [0,1,0]  # class 2
          ]
    :param y: un vecteur de labels
    :return:
    """
    # On crée une matrice de zéros ayant len(y) lignes et autant de colonnes que de classes distinctes
    one_hot_y = np.zeros((len(y), len(np.unique(y))))
    # Pour chaque ligne, on met un 1 à l'indice de la classe
    for ligne in range(len(one_hot_y)):
        one_hot_y[ligne][y[ligne]] = 1
    return one_hot_y


def g(X, theta):
    """
    Calcule la probabilité que chaque ligne de X corresponde à chaque classe
    :param X:
    :param theta:
    :return:
    """
    return softmax(np.dot(X, theta.T))


def E(X, y, theta):
    """
    Calcule la fonction de coût
    :param X: Données d'entrée
    :param y: Données de sortie
    :param theta: Paramètres du modèle
    :return: La fonction de coût
    """
    return -np.sum(y * np.log(g(X, theta))) / len(X)


def descente_gradient(X, y, theta, alpha=0.01, num_iters=10000):
    """
    Effectue la descente de gradient
    :param X:
    :param y:
    :param theta:
    :param alpha:
    :param num_iters:
    :return:
    """
    n = len(y)
    for i in range(num_iters):
        h = g(X, theta)
        theta -= alpha / n * np.dot((h - y).T, X)
    return theta
