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
    
    Cette fonction permet de récupérer les données utiles à l'évaluation du score
    
    :param filename: 
    :type filename:
    
    :return: l'auteur et la plante de l'image
    :rtype: string, string
     
    :Example:
        
        
    >>> 
    >>> 
     
    """

    tree = ET.parse(filename)
    root = tree.getroot()
    
    user=root[1][8].text
    observationId=root[1][0].text
    
    return user, observationId
    
    
def XMLmodif(author,plantID,filename):
    
    
    """
    
    Cete fonction permet de stocker dans un fichier xml les données nécessaires à
    
    :param author: nom de l'auteur de l'image
    :type author: string
    :param plantID: ID de la plante
    :type plantID: string
    :param img: nom de l'image associée aux donnée:
    :type img: string
    
    
     
    :Example:
        
        
    >>> 
    >>>
    
    """
    
    if not os.path.isfile('metrics.xml'):
        fo = open("metrics.xml", "wb")
        top = Element('authors')
    
    tree = ET.parse('metrics.xml')
    top = tree.getroot()
    
    #creation si non existence des elements
    if not top.find(str(author)):
        aut=Element(str(author)) 
        top.extend(aut)
    
    if not aut.find(str(plantID)):
        ID=Element(str(plantID))
        aut.extend(ID)
    
    if not ID.find('count'):
        count=Element('count')
        ID.extend(count)
    

    
    pictureID=Element(('pictureID'))
    ID.extend(pictureID)
    pictureID.text=filename
    count.text+=1
        
    
    fo.write(tree)
    fo.close()
    
    
    #solution alternative
    #author.set('name=', str(author))
    #plantID.set('name=', str(plantID))
    