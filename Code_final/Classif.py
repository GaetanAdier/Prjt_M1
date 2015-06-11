# -*- coding: utf-8 -*-
"""
This module is gathering all the fonctionnality conceived to process an image database classification.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

    
def K_means(Vectors, nb_centroid, iterat):
    
    """
    
    
    Function for computing the K-means method.
    
    This function find center of vectors and groups input samples
    
    
    This function is called as shown below :
    
    .. code-block:: python
       :emphasize-lines: 3,5
    
       [centroid_vec, val_dist] = K_means(all_desc, nb_word, 5)
    
    :param Vectors: The matrix which contains the whole descriptors of the database.
    :type Vectors: np.ndarray
    :param nb_centroid: Number of words you want.
    :type nb_centroid: int
    :param iterat: Number of iteration you want to find your centers.
    :type iterat: int 
    
    :return centroid_vec: centers of vectors 
    :rtype: nd.array
    :return val_dist: indices attribute to the vectors
    :rtype: nd.array
    

    """

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
    
    """
    
    
    Function for computing the Signature of the images.
    
    
    This function is called as shown below :
    
    .. code-block:: python
       :emphasize-lines: 3,5
    
       sig = Signature_img(all_desc, val_dist, nb_kp_per_img, nb_img, nb_word)
    
    :param Vectors: The matrix which contains the whole descriptors of the database.
    :type Vectors: np.ndarray
    :param val_dist: Matrix which contains the indices attribute to the vectors.
    :type val_dist: np.ndarray
    :param nb_kp_per_img: Number of Key-points per images.
    :type nb_kp_per_img: np.ndarray
    :param nb_img: Number of images in the database.
    :type nb_img: int
    :param nb_word: Number of words in the Bag of Words.
    :type nb_word: int 
    
    :return sig: Matric which contains all the signature of the database
    :rtype: np.ndarray
    
    """
    
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
    
    ur"""
    
    
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
    :param dType: The type of difference to compute, if = 0 => Euclidian distance, if = 1 => :math:`\chi^2`.
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
    
    