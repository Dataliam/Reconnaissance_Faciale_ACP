# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:41:59 2023

@author: liamb
"""

# Télécharger la base de données LFW
os.system("wget http://vis-www.cs.umass.edu/lfw/lfw.tgz")

# Décompresser la base de données
os.system("tar xvzf lfw.tgz")

# Supprimer le fichier compressé
os.system("rm lfw.tgz")








import cv2
import os

# Répertoire contenant les images
directory = 'C:/Users/liamb/Desktop/TER/data/Aaron_Eckhart'

# Liste pour stocker les images
images = []

# Boucle à travers toutes les images dans le répertoire
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Lire l'image
        img = cv2.imread(os.path.join(directory, filename))
        # Ajouter l'image à la liste
        images.append(img)
