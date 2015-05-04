# -*- coding: utf-8 -*-
"""
Created on Mon May 04 15:51:37 2015

@author: gaetan
"""

import os

##############################################################################
################################Function######################################
##############################################################################

def descript(path, name_desc, img):

    path_desc = "%s/%s" % (path, name_desc)    
    
    if not(os.path.isdir(path_desc)):
        os.mkdir(path_desc)
    
##############################################################################
################################Main##########################################
##############################################################################    
    
path = "C:\Users\gaetan\Documents\MASTER\Projet\directory"

if not(os.path.isdir(path)):
    os.mkdir(path)

descript(path, "tot")