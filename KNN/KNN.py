# -*- coding: utf-8 -*-
"""
Created on Mon Jun 08 16:48:24 2015

@author: etienne
"""
import operator

import numpy as np
from collections import Counter

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
        
test = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3,3,3,3,3,3,3,3],[6,6,6,6,6,6,6,6,6,6,6,6,6,6,6],[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9],[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]]
sig = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

classIds = [173, 173,173,6969,6969,6969,6969]


Test2 = KNN(test, classIds ,3,sig, 1)

print Test2
