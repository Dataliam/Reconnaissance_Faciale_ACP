# Importation des packages
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import re
import random
import shutil

# Chemin d'accès aux répertoires de travail
path = 'D:/TER'
os.chdir(path)

dossier_src = path + '/data2/' # Dossier source des image
dossier_dst = path + '/data_test/' # Dossier tempoaire pour stocké les images tests

##### Création du dossier test

# Créer une expression régulière pour faire correspondre le format du fichier
file_pattern = re.compile(r"\d+_(\d+)\.jpg")

# Parcourir les personnes
for person_id in range(1, 41):
    # Rechercher tous les fichiers correspondant au format "XXX_numéro.jpg" pour la personne actuelle
    matching_files = [f for f in os.listdir(dossier_src) if file_pattern.match(f) and int(file_pattern.match(f).group(1)) == person_id]

    # Choisir aléatoirement une image parmi les fichiers correspondants
    chosen_file = random.choice(matching_files)

    # Déplacer le fichier choisi vers le dossier de destination
    shutil.move(os.path.join(dossier_src, chosen_file),
                os.path.join(dossier_dst, chosen_file))
#####

##### Calcul des valeurs propres et vectreurs propres associés aux image du dossier source des images

# Importation de toutes les images et création d'une liste des noms
image_array = []
image_filenames = []

largeur_cible = 50
hauteur_cible = 50
taille_cible = (largeur_cible, hauteur_cible)

for filename in os.listdir(dossier_src):
    # Charger l'image
    image = Image.open(dossier_src + filename).convert("L")
    image_redimensionnee = image.resize(taille_cible)
    image_array.append(np.asarray(image_redimensionnee).flatten() / 255)
    image_filenames.append(dossier_src + filename)

# On empile les vecteurs pour former une matrice
matrice_image = np.vstack(image_array)

# Calcul des valeurs propres et vecteurs propres associés
valeurs_propres, vecteurs_propres = np.linalg.eigh(matrice_image.T @ matrice_image)

#####

##### Boucle for pour faire l'ACP sur toute les personnes du Dossier

for file in os.listdir(dossier_dst):
    # Obtenir le chemin complet du fichier
    file_path = os.path.join(dossier_dst, file)

    # Sortie d'une image test du répertoire
    test_image = Image.open(file_path).convert("L")
    test_image_redimensionnee = test_image.resize(taille_cible)

    # Transformer l'image en un vecteur de la même taille que les vecteurs d'image
    test_image_vector = np.asarray(
    test_image_redimensionnee, dtype=np.float32).flatten()

    # Normaliser le vecteur de l'image
    test_image_vector_normalise = test_image_vector / 255.0
    
    # Fonction pour calculer les distances euclidiennes et trouver la meilleure correspondance
    def trouver_meilleure_correspondance(num_composantes_principales, vecteurs_propres, matrice_image, test_image_vector_normalise):
        # Utiliser seulement un certain nombre de composantes principales
        vecteurs_propres_reduits = vecteurs_propres[:, -num_composantes_principales:]
        
        # Projeter les images sur les composantes principales sélectionnées
        images_projetees = vecteurs_propres_reduits.T @ matrice_image.T
        
        # Projeter l'image de test sur les composantes principales sélectionnées
        test_image_projetee = vecteurs_propres_reduits.T @ test_image_vector_normalise.reshape(-1, 1)
        
        # Calculer les distances euclidiennes entre le vecteur de l'image projeté et les vecteurs de caractéristiques principales des images
        
        distances = np.sqrt(np.sum((images_projetees - test_image_projetee) ** 2, axis=0))
        
        # Trouver l'indice de l'image la plus proche (i.e. la meilleure correspondance)
        meilleur_correspondance_indice = np.argmin(distances)
        
        return meilleur_correspondance_indice

    # Explorer différentes quantités de composantes principales
    # nombre arbitraire de vecteurs propres qu'on prend
    liste_num_composantes_principales = [1, 2, 3, 4, 5, 100, 1000, 2500]
    meilleures_correspondances = []

    for num_composantes_principales in liste_num_composantes_principales:
        meilleur_correspondance_indice = trouver_meilleure_correspondance(
        num_composantes_principales, vecteurs_propres, matrice_image, test_image_vector_normalise)
        meilleures_correspondances.append(meilleur_correspondance_indice)

    # Afficher les images de test et de correspondance pour chaque quantité de composantes principales
    fig, axes = plt.subplots(1, len(liste_num_composantes_principales) + 1, figsize=(15, 5))
    axes[0].imshow(test_image, cmap='gray')
    axes[0].set_title("Image de test")

    for i, meilleur_correspondance_indice in enumerate(meilleures_correspondances):
        meilleur_correspondance_image_nom_fichier = image_filenames[meilleur_correspondance_indice]
        meilleur_correspondance_image = Image.open(meilleur_correspondance_image_nom_fichier).convert("L")
        axes[i + 1].imshow(meilleur_correspondance_image, cmap='gray')
        axes[i + 1].set_title(f"{liste_num_composantes_principales[i]} composantes")

    plt.tight_layout()
    plt.show()

#####

##### Transfert du fichier de test vers le dossier source ##### Relancer cette section après chaque "bug"

for file in os.listdir(dossier_dst):
    # Obtenir le chemin complet du fichier
    file_path = os.path.join(dossier_dst, file)
    shutil.move(file_path, os.path.join(dossier_src, file))
    
#####


##### Modélisation de l'individu moyen

# Calculer l'individu moyen
vecteur_image_moyen = np.mean(matrice_image, axis=0)

# Transformer le vecteur moyen en image 2D
image_moyenne = vecteur_image_moyen.reshape(hauteur_cible, largeur_cible)

# Calculer l'individu moyen
vecteur_image_moyen = np.mean(matrice_image, axis=0)

# Transformer le vecteur moyen en image 2D
image_moyenne = vecteur_image_moyen.reshape(hauteur_cible, largeur_cible)

# Afficher l'individu moyen
plt.imshow(image_moyenne, cmap='gray')
plt.title("Individu moyen")
plt.show()

#####