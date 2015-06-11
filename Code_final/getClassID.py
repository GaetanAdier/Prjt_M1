# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:31:05 2015

@author: Projet
"""
import xml.etree.ElementTree as ET

def GetClassID(filename):
   
    """
    
    This function retrieves the ClassID from the metadata file.
    
    :param filename: nom du fichier XML
    :type filename:string
    
    :return: the category from the XML file
    :rtype: string
     
    :Example:
        
        
    >>> 
    >>> 
     
    """
    tree = ET.parse(filename)
    root = tree.getroot()    
    
    return root[4].text     

    

    