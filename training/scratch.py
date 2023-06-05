import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split


def scratch():
    scratch_path = "output/img/scratch"
    output = "output"

    # Chargement du dataset
    digits = datasets.load_digits()
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

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.savefig(f"{scratch_path}/matrice_confusion.pdf")
    plt.savefig(f"{scratch_path}/matrice_confusion.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:30], 'ro-', label='Vraies étiquettes')
    plt.plot(predictions[:30], 'bx-', label='Prédictions')
    plt.xlabel('Indices des exemples')
    plt.ylabel('Classe')
    plt.title('Comparaison des vraies étiquettes et des prédictions')
    plt.legend()
    plt.savefig(f"{scratch_path}/comparaison.pdf")
    plt.savefig(f"{scratch_path}/comparaison.png")
    plt.show()

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction, actual in zip(axes, X_test, predictions, y_test):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.set_title(f"Prediction: {prediction} ; Réelle: {actual}")
    plt.savefig(f"{scratch_path}/prediction.pdf")
    plt.savefig(f"{scratch_path}/prediction.png")
    plt.show()
