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
top = Element('root')

keypoints = [
    Element('keypoint',number=str(i))
    for i in xrange(nbKP-1)
    ]

top.extend(keypoints)

for i in range(nbKP-1):
    keypoint

print prettify(top)