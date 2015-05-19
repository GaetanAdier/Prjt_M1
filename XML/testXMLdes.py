# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:25:52 2015

@author: Projet
"""

import numpy as np
from xml.etree.ElementTree import Element, tostring
from xml.etree import ElementTree
from xml.dom import minidom

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
    
    
#nombre de key-points
nbKP=8

#
top = Element('top')

keypoints = [
    Element('kp')
    for i in xrange(nbKP-1)
    ]

top.extend(keypoints)


#--------------essais-------------------------------
#for kp0 in top.iter('kp0'):
#    kp0.text='0'

#for i in range(nbKP):
#    kp="kp"+str(i)
#    for kp in top.iter('kp'):
#        kp.text=0

#for k in root.iter('rank'):
#    rank.text = str(new_rank)
#tree.write('output.xml')
#

#for elem in top.iter():
#    elem.text=0


k=0
for kp in top:
#à changer pour que kp.text récupère 
    kp.text=str(k)
    k+=1

print prettify(top)