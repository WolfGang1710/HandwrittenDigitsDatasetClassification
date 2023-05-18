import os


def create_or_remove(filename, text=None):
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
