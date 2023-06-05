"""
@author : HERTZOG Thibaut & ROGUET William
"""

import os
import shutil
import time

import training.scratch as train_scratch
import training.scikit_learn as train_skt

print(f"==============================================\n"
      f"Classification de chiffres écrit à main levée.\n"
      f"==============================================\n"
      f"\n"
      f"Programme Python réalisé par Hertzog Thibaut & Roguet William.")

print(f"Initialisation")

folders = ['../output/', '../output/img', '../output/img/scikit-learn', '../output/img/scratch']

for folder in folders:
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"Contenu du dossier '{folder}' supprime.")
    os.makedirs(folder)
    print(f"Dossier '{folder}' cree.")

print(f"Termine.")

start = time.time()
train_skt.scikit_learn()
print(f"Temps d'exécution avec scikit-learn : {time.time()-start}")

start = time.time()
train_scratch.scratch()
print(f"Temps d'exécution avec la version scratch : {time.time()-start}")
