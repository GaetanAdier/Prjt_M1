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
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

#creation arbre XML pour le descripteur en input
def XMltree(des,name_desc):
    
    #nombre de key-points
    nbKP=np.size(des,0)
    
    #Balise racine avec nom du descripteur
    top = Element(str(name_desc))
    
    #Balises keypoints
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
    
    tree=prettify(top)
    
    print tree
    
    return tree

#creation du fichier xml avec arbre xml, chemin image et chemin dossier en input   
def XMLfile(tree,imgpath,path_desc):
    
    path=os.getcwd()
    
    os.chdir(path_desc)
    filename=ntpath.basename(imgpath)
    
    # creation fichier xml
    fo = open(filename+".xml", "wb")
    fo.write(tree);

    # Close opened file
    fo.close()
    
    os.chdir(path)
    