# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:50:51 2015

@author: gaetan
"""

import os
import glob
import cv2

from sphinx_doc import genere_doc
from sphinx_doc import configure_doc

def descript(path_work, name_desc, path_images, nb_images = "ALL", start_img = 1):
    
    """
    
    Cette fonction principale appelée dans le main qui permettra de créer les paths directory en fonction des différents descripteurs que l'utilisateur utilisera. Dans cette fonction nous retrouverons les paramètres suivant \:
    
     * **path_work** : Chemin où l'utlisateur souhaite créer ses dossiers pour les différents descripteurs.
     * **name_desc** : Nom du descripteur choisi.
     * **path_images** : Chemin où se trouve les images surlesquelles on vas travailler.
     * **nb_images** : Nombre d'images à traiter. Par défaut : ALL
     * **start_img** : Numéro de l'image de départ . Par défaut : 1
     
    """
    
    #creation du dossier contenant les donnnées de chaque images pour le descripteur choisi
    path_desc = "%s\\%s" % (path_work, name_desc) 
    
    if not(os.path.isdir(path_desc)):
        os.mkdir(path_desc)

    #creation d'une variable contenant les chemins de toutes les images contenues dans le dossier des images à traiter
    temp =  "%s\\*.jpg" % (path_images)
    list_path_img = glob.glob(temp)   
    nb_img = len(list_path_img)
    
    

def SIFT(img):
    
    """
    
    Cette fonction a pour but d'appliquer le calcul d'un descripteur SIFT à une image \:
    
     * **img** : Chemin de l'image que l'on souhaite traiter.
     * **des** : Valeurs retourner il s'agit d'une liste contenant les valeurs décrivant l'image
     
    """
    
    img_trait=cv2.imread(img)
    
    grayimage=cv2.cvtColor(img_trait, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayimage.jpg",grayimage)
    
    sift = cv2.SIFT()
    kp,des = sift.detectAndCompute(grayimage,None)
    
    return des

#genere_doc()