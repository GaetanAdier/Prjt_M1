# -*- coding: utf-8 -*-
"""
Created on Mon May 04 15:51:37 2015

@author: gaetan
"""

import os
import Process_flow as Pf
import matplotlib.pyplot as plt
from sphinx_doc import genere_doc
from sphinx_doc import configure_doc


##############################################################################
################################Main##########################################
##############################################################################    

path_images = "D:\\MASTER\\Projet\\analyse_res"   
path_work = "D:\\MASTER\\Projet\\directory"
descriptor = "SIFT"
nb_word = 100
sizeDesc = 128


#ID_classif, ID_connue, img = Pf.Process_flow(path_work, descriptor, path_images, nb_word, sizeDesc)



#configure_doc()
genere_doc()




