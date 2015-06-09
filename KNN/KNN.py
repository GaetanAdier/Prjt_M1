# -*- coding: utf-8 -*-
"""
Created on Mon Jun 08 16:48:24 2015

@author: etienne
"""
import operator

import numpy as np
from collections import Counter

def KNN(matSig, classId, k, Sig, dType):
    
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
sig = [9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]

classIds = [173, 173,173,6969,6969,6969,6969]


Test2 = KNN(test, classIds ,3,sig, 1)

print Test2
