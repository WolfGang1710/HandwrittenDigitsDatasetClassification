"""
@authors : HERTZOG Thibaut & ROGUET William
"""
import os  # Gestion de fichiers
import shutil  # Suppression de dossiers

from sklearn import datasets  # Chargement des données
from sklearn.model_selection import train_test_split  # Séparation des données en train et test
from method import *  # Méthodes de classification

print(f"==============================================\n"
      f"\tClassification de chiffres manuscrits.\n"
      f"==============================================\n"
      f"\n"
      f"Programme Python réalisé par Hertzog Thibaut & Roguet William.\n")

print(f"==============================================\n"
      f"\t\t\tInitialisation.\n"
      f"==============================================\n")

folders = ['img']

for folder in folders:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Contenu du dossier '{folder}' supprime.")
    os.makedirs(folder)
    print(f"Dossier '{folder}' cree.")

print(f"Termine.\n"
      f"==============================================\n")

print(f"Chargement des donneés.")
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size=0.2)

print(f"Termine.\n"
      f"==============================================\n")

print(f"Apprentissage avec la methode 'scikit-learn'.")
scikit_learn(X_train, X_test, y_train, y_test)

print(f"Apprentissage avec la methode 'from scratch'.")
scratch(X_train, X_test, y_train, y_test)