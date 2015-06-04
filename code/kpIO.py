# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 09:39:30 2015

@author: Projet
"""
import numpy as np
import os
import ntpath

def KPoutput(xydes,path_desc,imgpath):
   
    """
    
    Cette fonction permet de sauver les keypoints dans un fichier texte. Chaque
    ligne représente un keypoint sous la forme <x y des>
    
    :param xydes: coordonnées xy des kp
    :type xydes: float
    :param path_desc: chemin descripteur
    :type desc: string
    

     
    :Example:
        
        
    >>> 
    >>> 
     
    """
        
    
    
    #stockage du répertoire de travail actuel
    path=os.getcwd()
    
    #changement du répertoire de travail pour celui du descripteur utilisé
    os.chdir(path_desc)
    
    #récupération du nom du fichier image décrit
    filename=ntpath.basename(imgpath)
    
    
    # creation fichier xml
    np.savetxt(filename+'.txt', xydes)
    
    #retour à l'espace de travail précédent
    os.chdir(path)
    
    
def KPinput(filename):
    """
    
    Cette fonction permet de récupérer les keypoints sauvés
    
    :param xydes: coordonnées xy des kp
    :type xydes: float
    
    :param return: 
    :rtype:
     
    :Example:
        
        
    >>> 
    >>> 
     
    """
    
    data=np.genfromtxt(filename)
    
#    for i in xrange(len(data)):
#        x[i]=data[i][O]
#        y[i]=data[i][1]
#        des[i][2:end]=data[i][2:end]
        
    
    return data