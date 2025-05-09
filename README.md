# Handwritten Digits Dataset Classification
> HERTZOG Thibaut & ROGUET William

## Objectifs
Implémenter une méthode de régression logistique pour résoudre le problème de classication de chiffre écris à la main

## Dataset
Le dataset *handwritten digits dataset* provient de la bibliothèque `scikit-learn`. On peut le trouver [ici](https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset)

## Statégie
Le jeu de données contient 1797 entrées. Nous allons considérer $50%$ pour l'entrainement et $50%$ pour les tests 
## Partie 1 - A la main
* Considérer une partition de données ;
* Considérer un ensemble d'entraînement ;
* Construire un classifieur ;
* Considérer un ensemble de test ;
* Mesurer les performances obtenues.

## Partie 2 -  Avec `scikit-learn`
> Réaliser le même process avec les outils de `scikit-learn` et comparer les résultats obtenus avec ceux obtenus en [partie 1](#partie-1---a-la-main).

---

## ToDo
* [ ] Sur papier, décrypter les entrées, sorties, etc. du jeu de données ;
* [ ] Réaliser la partie 1 ;
* [x] Réaliser la partie 2 ;
* [ ] Comparer les résultats ;
* [x] Commenter le code et le rendre lisible (noms de variables, ...) ;
* [ ] Faire le rapport ;
* [ ] Corriger le rapport.

---

## Aborescence du projet

```bash
HandwrittenDigitsDatasetClassification
├── img # Image (et PDF) servant à comparer Scikit-learn avec notre version *from scratch*
├── ressources # Script que nous avons utilisé pour scikit-learn
│   └── plot_digits_classification.py
├── method # Les méthodes que nous avons implémenté
├── utils # Toutes fonctions utiles pour le projet
├── main.py # Script qui lancera ce présent dans `training`
└── README.md
```

## Dépendances

Merci de lancer la commande suivante dans un terminal : 
```bash
python -m pip install --upgrade pip ; pip install -r requirements.txt
```