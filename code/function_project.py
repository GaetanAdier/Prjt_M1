# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:50:51 2015

@author: gaetan
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ntpath

from sphinx_doc import genere_doc
from sphinx_doc import configure_doc


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
    
    sift = cv2.SIFT(0,3,0.04,10,1.6)
#    sift = cv2.SIFT()
    kp,des = sift.detectAndCompute(grayimage,None)
    
    return kp,des
    


def descript(path_work, name_desc, path_images, nb_word, nb_images = "ALL", start_img = 1):
    
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
        end_img = nb_img - start_img + 1
    else : 
        end_img = nb_images + start_img
    
    all_desc = np.zeros((0,128))
    nb_kp_per_desc = np.zeros((0,1))
    #application du descripteur choisit sur les images
    for i in range(start_img, (end_img + 1)): 
        kp,desc = SIFT(list_path_img[i-1])
        
    #Enregistrement des descripteurs dans fichiers txt
        filename=ntpath.basename(list_path_img[i-1])
        
        mat_kp=np.zeros([len(kp),130])  
        for i in range(len(kp)):
            mat_kp[i][0]=kp[i].pt[0]
        for j in range(len(kp)):
            mat_kp[j][1]=kp[j].pt[1]
        mat_kp[:,2:130]=desc
        
        np.savetxt(os.path.join(path_desc,filename+'.txt'),mat_kp,fmt='%f')
    
        rows, cols = desc.shape
        nb_kp_per_desc = np.append(nb_kp_per_desc, rows)
        all_desc = np.concatenate((all_desc, desc))
    
    [centroid_vec, val_dist] = K_means(all_desc, nb_word, 5)
  #  return K_means(all_desc, 10, 5)
    plt.hist(val_dist, np.arange(nb_word+1))
    plt.show()
    return val_dist 
    
    
def K_means(Vectors, nb_centroid, iterat):

    #création des vecteurs de manières aléatoires
    rows, cols = Vectors.shape
    val_dist = np.zeros((nb_centroid,rows))
    val_min = np.zeros(rows)
    centroid_vectors = np.zeros((nb_centroid, cols))
    centroid_vectors = [np.random.rand(1, cols)*255 for i in range(nb_centroid)]
    centroid_vectors = np.asarray(centroid_vectors)
    centroid_vectors = centroid_vectors.reshape(nb_centroid, cols)
    

    for it in range(iterat):
        for k in range(nb_centroid):
            for j in range(rows):
                val_dist[k,j] = (np.sqrt(sum((centroid_vectors[k, :]- Vectors[j, :])**2)))
                
                #dernière boucle 
                if k == nb_centroid-1:
                    val_min[j] = val_dist[k,j]
                    for i in range(nb_centroid):
                        if val_dist[i,j] <= val_min[j]:
                            val_min[j] = i
                
        for k in range(nb_centroid):
            nb = 1
            for j in range(rows):
                if val_min[j] == k :
                    centroid_vectors[k, :] = centroid_vectors[k, :] + Vectors[j, :]
                    nb = nb + 1
            centroid_vectors[k, :] = centroid_vectors[k, :]/nb
            
        return centroid_vectors, val_min

#def Signature_img(Vectors, k_means):
    
    

#genere_doc()