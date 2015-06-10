# -*- coding: utf-8 -*-
"""
Created on Mon May 04 15:51:37 2015

@author: gaetan
"""

import os
import function_project as fc
import matplotlib.pyplot as plt
from sphinx_doc import genere_doc
from sphinx_doc import configure_doc


##############################################################################
################################Main##########################################
##############################################################################    

path_images = "D:\\MASTER\\Projet\\analyse_res"   
path_work = "D:\\MASTER\\Projet\\directory"
descriptor = "C2O"
nb_word = 100
sizeDesc = 72
if not(os.path.isdir(path_work)):
    os.mkdir(path_work)
    
if not(os.path.isdir(path_images)):
    print("path for work on images doesn't exist")
    exit(0)

ID_classif, ID_connue, img = fc.descript(path_work, descriptor, path_images, nb_word, sizeDesc)













