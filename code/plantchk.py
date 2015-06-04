# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 14:49:51 2015

@author: Projet
"""
import numpy as np

def plantCHK():
    
    """
    
    Cette fonction vérifie l'exactitude des plantes supposées et retourne seulement les réponses correctes.
    
    
    :param :
    :type :
    """
#à faire récupération des classID ordonnées dans le sens du run. plutot utiliser fichier run avec     
    
    
    #rappel format run :<ObservationId;ClassId;rank;score>    
    #données du run
    hyp=np.loadtxt('fakerun.txt',delimiter=';')
    
    chkd=np.empty() 
    #vérification des hypothèses
    for i in xrange(len(hyp)):
        #conservation prédictions correctes         
        if (hyp[i][3] == classID[i]):
            try:
                chkd=np.vstack(chkd,hyp[i][:])
            except NameError:
                chkd=hyp[i][:]

        #conservation prédictions erronées pour analyse
        else:
            try:
                wrng=np.vstack(wrng,hyp[i][:])
            except NameError:
                wrng=hyp[i][:]
           
            
            
            
    return chkd