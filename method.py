import seaborn
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

from utils import *


def scikit_learn(X_train, X_test, y_train, y_test):
    """
    Méthode de classification avec scikit-learn
    :param X_train: Données d'entrainement (entrées)
    :param X_test:  Données de test (entrées)
    :param y_train: Données d'entrainement (sorties)
    :param y_test: Données de test (sorties)
    :return: None
    """
    # Création du modèle
    model = LogisticRegression(max_iter=10000)
    # Apprentissage
    model.fit(X_train, y_train)
    # Prédiction
    predicted = model.predict(X_test)

    # Affichage des 4 premières images ainsi que leur label
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction, actual in zip(axes, X_test, predicted, y_test):
        ax.set_axis_off()
        image = image.reshape(8, 8)  # 8x8 pixels
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.set_title(f"Prediction: {prediction} ; Réelle: {actual}")
    plt.savefig("img/prediction_sklearn.pdf")
    plt.savefig("img/prediction_sklearn.png")
    plt.show()
    plt.close()

    cm = confusion_matrix(y_test, predicted)
    # Affichage de la matrice de confusion
    # Nous avons demandé à chat-gpt car la méthode avec
    # plt était trop artisanale à notre goût.
    seaborn.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Prédiction")
    plt.ylabel("Réelle")
    plt.savefig("img/confusion_matrix_sklearn.pdf")
    plt.savefig("img/confusion_matrix_sklearn.png")
    plt.show()

    accuracy = metrics.accuracy_score(y_test, predicted)
    print(f"Accuracy avec scikit-learn sur le jeu de test: {accuracy:.4f}")
    predicted = model.predict(X_train)
    accuracy = metrics.accuracy_score(y_train, predicted)
    print(f"Accuracy avec scikit-learn sur le jeu d'entrainement: {accuracy:.4f}")


def scratch(X_train, X_test, y_train, y_test):
    y_train_encoded = transform_labels(y_train)

    # Ajout d'une colonne de biais à X_train
    X_train_ones = np.column_stack((np.ones(len(X_train)), X_train))

    # Initialisation de theta
    theta_init = np.zeros((len(np.unique(y_train)), X_train_ones.shape[1]))

    # Entraînement du modèle
    theta = descente_gradient(X_train_ones, y_train_encoded, theta_init)

    # Test du modèle
    X_test_ones = np.column_stack((np.ones(len(X_test)), X_test))
    X_train_ones = np.column_stack((np.ones(len(X_train)), X_train))
    # On récupère l'indice de la classe prédite ayant la plus grande probabilité
    # pour chaque image
    y_pred_test = np.argmax(g(X_test_ones, theta), axis=1)
    y_pred_train = np.argmax(g(X_train_ones, theta), axis=1)

    # Calcul de l'accuracy sur le test
    accuracy = accuracy_score(y_test, y_pred_test)
    print("Accuracy sur le jeu de test:", accuracy)

    # Calcul de l'accuracy sur le train
    accuracy = accuracy_score(y_train, y_pred_train)
    print("Accuracy avec sur le jeu d'entrainement:", accuracy)

    # Affichage des 4 premières images ainsi que leur label
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction, actual in zip(axes, X_test, y_pred_test, y_test):
        ax.set_axis_off()
        image = image.reshape(8, 8)  # 8x8 pixels
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.set_title(f"Prediction: {prediction} ; Réelle: {actual}")
    plt.savefig("img/prediction_scratch.pdf")
    plt.savefig("img/prediction_sratch.png")
    plt.show()
    plt.close()

    cm = confusion_matrix(y_test, y_pred_test)
    seaborn.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Prédiction")
    plt.ylabel("Réelle")
    plt.title("Matrice de confusion sur le jeu de test")
    plt.savefig("img/confusion_matrix_scratch_test.pdf")
    plt.savefig("img/confusion_matrix_scratch_test.png")
    plt.show()

    cm = confusion_matrix(y_train, y_pred_train)
    seaborn.heatmap(cm, annot=True, cmap="Blues")
    plt.xlabel("Prédiction")
    plt.ylabel("Réelle")
    plt.title("Matrice de confusion sur le jeu d'entrainement")
    plt.savefig("img/confusion_matrix_scratch_train.pdf")
    plt.savefig("img/confusion_matrix_scratch_train.png")
    plt.show()
