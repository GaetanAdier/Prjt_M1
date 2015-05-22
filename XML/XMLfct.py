# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:23:05 2015

@author: Projet
"""

import os
import ntpath
import numpy as np
from xml.etree.ElementTree import Element
from xml.etree import ElementTree
from xml.dom import minidom




#fonction permettant d'avoir en sortie un arbre XML proprement mis en page
def prettify(elem):
    
    """
    Return a pretty-printed XML string for the Element.
    
    :param elem: matrice contenant les vecteurs keypoints obtenus par le descripteur
    :type elem: Element (structure importé d'xml.etree.ElementTree)
    
    :return: l'arbre XML mis en page
    :rtype: string
    
    :Example:
        
        
    >>>tree=prettify(top) 
    >>>print tree 
    """
    
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

#creation arbre XML pour le descripteur en input
def XMltree(des,name_desc):
    
    """
    Creation de l'arbre du XML à partir des

    :param des: matrice contenant les vecteurs keypoints obtenus par le descripteur
    :type des: array
    :param name_desc: nom du descripteur utilisé
    :type name_desc: string
    
    :return: l'arbre XML 
    :rtype: string
    
    :Example:
        
        
    >>> arbre=XMLtree(a,sift)
    >>> 
    """    
    
    
    #nombre de key-points
    nbKP=np.size(des,0)
    
    #Balise racine avec nom du descripteur
    top = Element(str(name_desc))
    
    # Création des balises keypoints selon le nmbre 
    keypoints = [
        Element('keypoint', num=str(i))
        for i in xrange(nbKP-1)
        ]
    
    top.extend(keypoints)
    
    k=0
    for kp in top:
        #à modifier selon format descripteur reçu
        kp.text=des[k,:]
        k+=1
    
    #envoi de l'arbre vers la fonction prettify pour le rendre plus lisible
    tree=prettify(top)
    
    #print test
    print tree
    
    return tree

#creation du fichier xml avec arbre xml, chemin image et chemin dossier en input   
def XMLfile(tree,imgpath,path_desc):
    
    """
    Creation du fichier XML dans le dossier du descripteur

    :param tree: matrice contenant les vecteurs keypoints obtenus par le descripteur
    :type tree: string
    :param imgpath: chemin du fichier image 
    :type denominateur: string
    :param : chemin du dossier du descripteur utilisé
    :type denominateur: string
    
    :Example:
        
        
    >>>arbre=XMLtree(a,sift)
    >>>XMLfile(arbre, C:\Projet\test.jpeg, C:\Projet\SIFT\)
   
    """
    
    #stockage du répertoire de travail actuel
    path=os.getcwd()
    
    #changement du répertoire de travail pour celui du descripteur utilisé
    os.chdir(path_desc)
    
    #récupération du nom du fichier image décrit
    filename=ntpath.basename(imgpath)
    
    # creation fichier xml
    fo = open(filename+".xml", "wb")
    fo.write(tree)

    # Close opened file
    fo.close()
    
    #retour à l'espace de travail précédent
    os.chdir(path)
    
