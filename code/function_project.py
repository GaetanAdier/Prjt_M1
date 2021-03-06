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
import random
from collections import Counter

import getClassID as gtID
import C2O as CO

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
    


def descript(path_work, name_desc, path_images, nb_word, sizeDesc, nb_images = "ALL", start_img = 1):
    
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
    temp2 =  "%s\\*.xml" % (path_images)
    list_path_img = glob.glob(temp)   
    list_path_img2 = glob.glob(temp2)   
    nb_img = len(list_path_img)
    
    #permet de parcourir toutes ou un nombre d'image définis par l'utilisateur de manière automatique    
    if(nb_images == "ALL"):
        end_img = nb_img - start_img + 1
    else : 
        end_img = nb_images + start_img
    
    all_desc = np.zeros((0,sizeDesc))
    nb_kp_per_img = np.zeros((0,1))
    sig = np.zeros((0,nb_word)) 
    
    ID = np.zeros(0)
    #application du descripteur choisit sur les images
    for i in range(start_img, (end_img + 1)): 
        #kp,desc = SIFT(list_path_img[i-1])
        mat_kp, desc = CO.C2OPatch(list_path_img[i-1], 4, 6, 3)
    
        temp =  gtID.GetClassID(list_path_img2[i-1])
        ID = np.append(ID, temp)
        print ID
        
        
    #Enregistrement des descripteurs dans fichiers txt
        filename=ntpath.basename(list_path_img[i-1])
        
#        mat_kp=np.zeros([len(kp),(sizeDesc+2)])  
#        for i in range(len(kp)):
#            mat_kp[i][0]=kp[i].pt[0]
#        for j in range(len(kp)):
#            mat_kp[j][1]=kp[j].pt[1]
#        mat_kp[:,2:(sizeDesc+2)]=desc
        
        np.savetxt(os.path.join(path_desc,filename+'.txt'),mat_kp,fmt='%f')
    
        rows, cols = desc.shape
        nb_kp_per_img = np.append(nb_kp_per_img, rows)
        all_desc = np.concatenate((all_desc, desc))

    
    [centroid_vec, val_dist] = K_means(all_desc, nb_word, 5)
    sig = Signature_img(all_desc, val_dist, nb_kp_per_img, nb_img, nb_word)
    
    sigTrain = sig[0:69]  
    sigTest = sig[70:99]
    res = np.zeros(0)
    
    for i in range(len(sigTest)):
        test = KNN(sigTrain, ID, 5, sigTest[i], 0)
        res = np.append(res, test)
    
    return res, ID[70:99], list_path_img
    
    
def K_means(Vectors, nb_centroid, iterat):

    #création des vecteurs de manières aléatoires
    rows, cols = Vectors.shape
    val_dist = np.zeros((nb_centroid,rows))
    val_min = np.zeros(rows)
    ind_min = np.zeros(rows)
    centroid_vectors = np.zeros((nb_centroid, cols))
    
    for i in range(nb_centroid):
        centroid_vectors[i, :] = Vectors[random.randint(0, rows), :]
    

    for it in range(iterat):
        for k in range(nb_centroid):
            for j in range(rows):
                val_dist[k,j] = (np.sqrt(sum((centroid_vectors[k, :]- Vectors[j, :])**2)))
                
                #dernière boucle 
                if k == nb_centroid-1:
                    val_min[j] = val_dist[k,j]
                    for i in range(nb_centroid):
                        if val_dist[i,j] <= val_min[j]:
                            val_min[j] = val_dist[i,j]
                            ind_min[j] = i
                            
                
        for k in range(nb_centroid):
            nb = 1
            for j in range(rows):
                if val_min[j] == k :
                    centroid_vectors[k, :] = centroid_vectors[k, :] + Vectors[j, :]
                    nb = nb + 1
            centroid_vectors[k, :] = centroid_vectors[k, :]/nb
            
        return centroid_vectors, ind_min

def Signature_img(Vectors, val_dist, nb_kp_per_img, nb_img, nb_word):
    
    start = 0
    end = 0
    sig = np.zeros((0,nb_word))
    test = np.zeros((0,nb_word))
    for i in range(nb_img): 
        end = end + nb_kp_per_img[i]
        [test, bins, patches] = plt.hist(np.array(val_dist[start:end]), nb_word)
        test = np.resize(test, (1, nb_word))
        sig = np.concatenate((sig, test))
        start = end
    
    return sig
    
def KNN(matSig, classId, k, Sig, dType):
    
    """
    
    
    Function for computing the K-nn method.
    
    This function takes as argument the dictionarry of know signatures associate with their classID, and the signature of the image to classify
    

    
    This function is called as shown below :
    
    .. code-block:: python
       :emphasize-lines: 3,5
    
       ClassId = KNN(matSig, classId ,k,Sig, dType)
    
    :param matSig: The matrix which contains the dictionnary of know signature.
    :type matSig: np.ndarray
    :param classId: The array containing the class ids which corresponds to the signatures in matSig.
    :type classId: np.ndarray
    :param k: Number of nearest neightbor to keep for the class attribution.
    :type k: int
    :param Sig: The array which contains the signature of the image to classify.
    :type Sig: np.ndarray
    :param dType: The type of difference to compute, if = 0 => Euclidian distance, if = 1 => :math:`\khi^2`.
    :type dType: int    
    
    :return: The classID to attribute for the image to classify
    :rtype: float
    

    """
    
    matSig = np.array(matSig)
    Sig = np.array(Sig)
    
    matDiff = np.zeros(np.shape(matSig))    
    
    
    if dType == 0:     
        matDiff = np.sqrt((matSig - Sig)**2)
        matDiff = np.sum(matDiff,axis=1)
    else:
        matDiff = np.sum((matSig-Sig)**2/(matSig+Sig+0.0000001)**2,axis=1)

        
    
    index = np.argsort(matDiff, axis=-1, kind='quicksort', order=None)
    

    classIdn = list(np.zeros(k))
    
    
    for i in np.arange(0,k):
        classIdn[i] = classId[index[i]]
        
    
    
    Occurence = Counter(classIdn)
    
    Classe = Occurence.most_common(1)


    ClassToAffect = Classe[0][0]
    
    return ClassToAffect    
    
    

#genere_doc()