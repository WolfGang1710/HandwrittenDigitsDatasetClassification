import os


def create_or_remove(filename, text=None):
    """
    Supprime le fichier `filename` s'il existe, et en créé un nouveau en écrivant `text` dedans
    si `text` est passé en paramètre
    :param filename: (str) nom du fichier avec l'extension
    :param text: (str) texte à écrire dans le fichier
    """
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Fichier '{filename}' supprimé.")

    if text is not None:
        with open(filename, 'w') as file:
            file.write(text)
        print(f"Fichier '{filename}' créé et le texte a été écrit.")
    else:
        with open(filename, 'w') as file:
            file.write('')
        print(f"Fichier '{filename}' créé.")
