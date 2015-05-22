# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:46:42 2015

@author: Projet
"""
import XMLparser
import os

def metrics(p):
    
    """
    
    Cete fonction permet d'évaluer le score de la solution
    
    :param Users: number of users who have at least one image in the test data
    :type Users:
    :param plants: number of plants observed by the u-th user
    :type plants:
    :param Sc: score between 1 and 0 equals to the inverse of the rank of the correct species (for the n-th picture taken from the p-th plant observed by the u-th user)
    :type Sc:
    :param N: number of pictures taken from the p-th plant observed by the u-th user
 
     
    :Example:
        
        
    >>> 
    >>> 
     
    """
    
    #Primary metric
    #Average classification score
    S1=1/Users * sum(1/P[u] * sum(Sc[u][p],1),1)
    
    #Secondary metric
    S2=S=1/Users * sum(1/P[u] * sum(1/N * sum(Sc[u][p],1),1),1)
    

def MetricsArgs(IBPath):
    
   
    """
    
    Cette fonction permet de récupérer tous les arguments nécessaires à l'évaluation. 
    Elle itère tous les fichiers xml à travers le dossiers de la banque d'image afin de 
    récupérer l'auteur et la plante de chaque image
    
    :param Users: number of users who have at least one image in the test data
    :type Users:

 
     
    :Example:
        
        
    >>> 
    >>> 
     
    """
    
    machin=zeros() 
    
    
    #iteration sur tous les fichiers XML de la banque d'images
    for i in os.listdir(IBPath):
        if i.endswith(".xml"): 
            
            [user, ObsID]=XMLparser(i+'.xml')
            
            while i<len(machin):
                
                if user==machin[i][j]:
                    
                    machin[i][j]+=1
                    
                else:
                    
            
            
            
            continue
        
        else:
            continue
    
    