# -*- coding: utf-8 -*-
"""
Created on Wed May 13 10:50:51 2015

@author: gaetan
"""

import os
from sphinx_doc import genere_doc
from sphinx_doc import configure_doc

def descript(path_work, name_desc, path_images, nb_images = "ALL", start_img = 1):
    
    """
    
    Cette fonction principale appelée dans le main qui permettra de créer les paths directory en fonction des différents descripteurs que l'utilisateur utilisera. Dans cette fonction nous retrouverons les paramètres suivant \:
    
     * **path_work** : Chemin où l'utlisateur souhaite créer ses dossiers pour les différents descripteurs.
     * **name_desc** : Nnom du descripteur choisi.
     * **path_images** : Chemin où se trouve les images surlesquelles on vas travailler.
     * **nb_images** : Nombre d'images à traiter. Par défaut : ALL
     * **start_img** : Numéro de l'image de départ . Par défaut : 1
     
    """
    
    path_desc = "%s/%s" % (path_work, name_desc)    
    
    if not(os.path.isdir(path_desc)):
        os.mkdir(path_desc)

genere_doc()