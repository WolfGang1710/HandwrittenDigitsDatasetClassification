import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Chargement du dataset
digits = load_digits()
X = digits.data
y = digits.target

# Prétraitement des données
X /= 16.0  # Mise à l'échelle des valeurs des pixels entre 0 et 1

# Séparation des données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)


# Fonction d'erreur : entropie croisée (cross-entropy)
def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-6  # Petite valeur ajoutée pour éviter les divisions par zéro
    num_samples = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + epsilon)) / num_samples
    return loss


# Fonction softmax
def softmax(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# Initialisation des poids et du biais
num_features = X_train.shape[1]
num_classes = len(np.unique(y_train))
weights = np.random.randn(num_features, num_classes)
bias = np.zeros((1, num_classes))

# Paramètres d'apprentissage
learning_rate = 0.1
num_iterations = 1000

# Entraînement du modèle par descente de gradient
for i in range(num_iterations):
    # Calcul des scores
    scores = np.dot(X_train, weights) + bias
    # Calcul des probabilités avec la fonction softmax
    probabilities = softmax(scores)

    # Calcul du gradient de la fonction d'erreur
    gradient = probabilities - np.eye(num_classes)[y_train]

    # Mise à jour des poids et du biais
    weights -= learning_rate * np.dot(X_train.T, gradient)
    bias -= learning_rate * np.sum(gradient, axis=0)

    # Calcul de l'erreur
    loss = cross_entropy_loss(np.eye(num_classes)[y_train], probabilities)

    if (i + 1) % 100 == 0:
        print(f"Iteration {i + 1}/{num_iterations}, Loss: {loss:.4f}")

# Prédiction sur l'ensemble de test
test_scores = np.dot(X_test, weights) + bias
test_probabilities = softmax(test_scores)
predictions = np.argmax(test_probabilities, axis=1)

# Évaluation des performances
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.4f}")
