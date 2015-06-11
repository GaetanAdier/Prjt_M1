# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 16:43:07 2015

@author: Projet
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:08:40 2015

@author: Projet
"""
import os.path
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element

def XMLparser(filename):
   
    """
    This function scan through the metadata 
    Cette fonction récupère scan le fichier XML en entrée et renvoie les 
    données utiles qui sont : la classID, l'auteur de l'image et l'observationID
    associés au nom du fichier image.
    
    :param filename: 
    :type filename:
    
    :return: l'auteur et la plante de l'image
    :rtype: string, string
     
    :Example:
        
        
    >>>[user,observationID,classID]=XMLparser("D:\\MASTER\\Projet\\traintest") 
    >>> 
     
    """

    tree = ET.parse(filename)
    root = tree.getroot()
    
    user=root[8].text
    observationId=root[0].text
    classID=root[4].text
    
    return user, observationId, classID
    
    
def metaTXT(IBpath):
    
    
    """
    
    Cette fonction parcourt la banque d'images afin de stocker les métadonnées nécessaires dans 
    :param IBpath: nom de l'auteur de l'image
    :type IBpath: string

    
    
     
    :Example:
        
        
    >>>metaTXT("D:\\MASTER\\Projet\\traintest") 
    >>>
    
    """
    
    
    for i in os.listdir(IBpath):
        if i.endswith(".xml"): 
            
            
            [user,observationID,classID]=XMLparser(i)
            #filename;author;classID;observationID
            line='%s;%s;%s;%s;%s'%(user, observationID, classID, i)
            
            with open('meta.txt', 'a') as file:
                file.write(line)

    