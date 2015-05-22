# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:50:51 2015

@author: gaetan
"""

import os
import glob
import cv2
import numpy as np

from sphinx_doc import genere_doc
from sphinx_doc import configure_doc

def descript(path_work, name_desc, path_images, nb_images = "ALL", start_img = 1):
    
    """
    
    Cette fonction principale appelée dans le main qui permettra de créer les paths directory en fonction des différents descripteurs que l'utilisateur utilisera. Dans cette fonction nous retrouverons les paramètres suivant \:
    
    :param path_work: Chemin où l'utlisateur souhaite créer ses dossiers pour les différents descripteurs.
    :type path_work: string
    :param name_desc: Nom du descripteur choisi.
    :type name_desc: string
    :param path_images: Chemin où se trouve les images surlesquelles on vas travailler.
    :type path_images: string
    :param nb_images: Nombre d'images à traiter. Par défaut : ALL
    :type nb_images: int
    :param start_img: Numéro de l'image de départ . Par défaut : 1
    :type start_img: int
     
    :Example:
        
        
    >>> 
    >>> 
     
    """
    
    #creation du dossier contenant les donnnées de chaque images pour le descripteur choisi
    path_desc = "%s\\%s" % (path_work, name_desc) 
    
    if not(os.path.isdir(path_desc)):
        os.mkdir(path_desc)

    #creation d'une variable contenant les chemins de toutes les images contenues dans le dossier des images à traiter
    temp =  "%s\\*.jpg" % (path_images)
    list_path_img = glob.glob(temp)   
    nb_img = len(list_path_img)
    
    #permet de parcourir toutes ou un nombre d'image définis par l'utilisateur de manière automatique    
    if(nb_images == "ALL"):
        end_img = nb_img - start_img
    else : 
        end_img = nb_images + start_img
        
    #application du descripteur choisit sur les images
    for i in range(start_img, (end_img + 1)): 
        SIFT(list_path_img[i-1])
        
    
    
def SIFT(img):
    
    """
    
    Cette fonction a pour but d'appliquer le calcul d'un descripteur SIFT à une image \:
    
    :param img: Chemin de l'image que l'on souhaite traiter.
    :type img: string
    :param des: Valeurs retourner il s'agit d'une liste contenant les valeurs décrivant l'image
    :type des: array
    
    :return: 
    :rtype:
     
    :Example:
        
        
    >>> 
    >>>
    """
    
    img_trait=cv2.imread(img)
    
    grayimage=cv2.cvtColor(img_trait, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayimage.jpg",grayimage)
    
    sift = cv2.SIFT()
    kp,des = sift.detectAndCompute(grayimage,None)
    
    return des
    
def K_means():

    #création des vecteurs de manières aléatoires
    val_dist = np.zeros((3,100))
    vectors_test = np.zeros((100, 128))
    centroid_vectors = np.zeros((3, 128))
    vectors_test = [np.random.rand(1, 128)*255 for i in range(100)]
    centroid_vectors = [np.random.rand(1, 128)*255 for i in range(3)]
    vectors_test = np.asarray(vectors_test)
    vectors_test = vectors_test.reshape(100, 128)
    centroid_vectors = np.asarray(centroid_vectors)
    centroid_vectors = centroid_vectors.reshape(3, 128)
    
    
    for k in range(3):
        val_dist[k,:] = (np.sqrt(sum((centroid_vectors[k, :]- vectors_test[np.arange(0,99), :])**2)))
        
    
    
            
    

#genere_doc()