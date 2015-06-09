# -*- coding: utf-8 -*-
"""
Created on Mon Jun 08 16:48:24 2015

@author: etienne
"""

import numpy as np
from collections import Counter

def KNN(matSig, classId ,k,Sig):
    
        
    
    matDiff = np.zeros(np.shape(matSig))    
    
    matDiff = np.mean(matSig**2 - Sig**2)
    
    
    index = np.argsort(matDiff, axis=-1, kind='quicksort', order=None)
    

    matDiff = matDiff[index]
    classId = classId[index]
    
    classId = classId[0:k]
    
    Occurence = Counter(classId)
    
    ClassToAffect = Occurence.most_common(1)
    
    #Contient le string du nom de la classe a affecter a l'image
    ClassToAffect = ClassToAffect[0][0]