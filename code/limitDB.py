# -*- coding: utf-8 -*-
"""
Created on Tue Jun 09 09:16:50 2015

@author: Projet
"""


import os.path
import xml.etree.ElementTree as ET



def limitDB(ID,DBpath,category='species'):
   
    """
    
    This function is used to limit the database to a specified specy, genus, family or content. It scans through the metadata of the xml files related to each picture and return a list containing all the filenames corresponding to the wanted category.
    

    :param ID: ID specifying the wanted category
    :type ID: string
    :param DBpath: Path of the image DataBase
    :type DBpath: string
    :param category: category which will limit the database. The available choices are : content(Branch, Entire, Flower, Fruit, Leaf, LeafScan, Stem), species, genus, family
    :type category: string
    
    :return: list containing all the filenames representing the wanted plant
    :rtype: List
     
    :Example:
        
        
    >>>limitedDB("Flower","D:\\MASTER\\Projet\\traintest",content) 
    >>>limitedDB("Branch","D:\\MASTER\\Projet\\traintest")  
     
    """
    fileList=[]
    
    if category == "content":
        indice=3
    elif category == "species":
        indice=7
    elif category == "genus":
        indice=6
    elif category == "family":
        indice=5

    
    for i in os.listdir(DBpath):
        if i.endswith(".xml"):
            tree = ET.parse(DBpath + "\\" + i)
            root = tree.getroot()
            
            
            if root[indice].text == ID:
                if not fileList:
                    fileList=[i+"jpg"]
                else:
                    fileList += [i+".jpg"]
        
        print i
                
                
             
    
    return fileList
    
test = limitDB("Picris hieracioides L.", "D:\\MASTER\\Projet\\train")
print test