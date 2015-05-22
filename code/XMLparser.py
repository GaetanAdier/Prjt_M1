# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:08:40 2015

@author: Projet
"""

import xml.etree.ElementTree as ET

def XMLparser(filename):
   
    """
    
    Cete fonction permet de récupérer les données utiles à l'évaluation du score
    
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