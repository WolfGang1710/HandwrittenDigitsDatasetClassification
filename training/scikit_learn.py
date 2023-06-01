"""Classification des chiffres écrit à la main avec scikit-learn.

Ce script utilise uniquement les fonctions offertes par Scikit-learn.
Aucune fonction `fait maison' n'est utilisée.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.file import create_or_remove


def scikit_learn():
    scikit_learn_path = "../output/img/scikit-learn"
    output = "../output"

    # Chargement du dataset
    digits = datasets.load_digits()

    # Affichage des 4 premières images ainsi que leur label
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))

    # Construction d'un itérable composé de `axes`, `digits.images` et  `digits.target`
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.set_title("Training: %i" % label)
    plt.savefig(f"{scikit_learn_path}/visualisation.pdf")
    plt.savefig(f"{scikit_learn_path}/visualisation.png")
    plt.show()

    ###############################################################################
    # Classification
    # --------------

    # Applatissement des images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Classifier
    model = LogisticRegression(max_iter=10000)

    # Split data : 50% train et : 50% test
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

    # Apprentissage
    model.fit(X_train, y_train)

    # Prédiction
    predicted = model.predict(X_test)

    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap="gray", interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
    plt.savefig(f"{scikit_learn_path}/prediction.pdf")
    plt.savefig(f"{scikit_learn_path}/prediction.png")
    plt.show()

    report = metrics.classification_report(y_test, predicted)
    print(
        f"Classification report for classifier {model}:\n"
        f"{report}\n"
    )

    create_or_remove(f"{output}/report.txt", report)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.savefig(f"{scikit_learn_path}/matrice_confusion.pdf")
    plt.savefig(f"{scikit_learn_path}/matrice_confusion.png")
    plt.show()
