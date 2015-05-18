# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:23:05 2015

@author: Projet
"""
import numpy as np
from xml.etree.ElementTree import Element, tostring
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
def XMltree(des):
    
    #nombre de key-points
    nbKP=np.size(des,0)
    
    #Balise racine
    top = Element('root')
    
    #Balises keypoints
    keypoints = [
        Element('keypoint', num=str(i))
        for i in xrange(nbKP-1)
        ]
    
    top.extend(keypoints)
    
    tree=prettify(top)
    
    print tree
    
    return tree

#creation du fichier xml pour l'arbre en input    
def XMLfile(tree,num):
    
    # creation fichier xml
    fo = open(num+".xml", "wb")
    fo.write(tree);

    # Close opend file
    fo.close()
    