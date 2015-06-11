# -*- coding: utf-8 -*-
"""
This module provide a function  to get the class ID from the CLEF metadatas files

"""
import xml.etree.ElementTree as ET

def GetClassID(filename):
   
    """
    
    This function retrieves the ClassID from the metadata file.
    
    :param filename: nom du fichier XML
    :type filename: string
    
    :return: the category from the XML file
    :rtype: string
     

     
    """
    tree = ET.parse(filename)
    root = tree.getroot()    
    
    return root[4].text     

    

    