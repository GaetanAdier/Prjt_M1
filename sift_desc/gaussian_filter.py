# -*- coding: utf-8 -*-
"""
Created on Thu Jun 04 10:44:19 2015

@author: RTMA
"""

import cv2
import numpy as np

#from sphinx_doc import genere_doc
#from sphinx_doc import configure_doc

def gauss(img):
    
    
    """
    Cette fonction permet de réaliser l'opération de gradient pour lisser l'image 
    dans l'algorithme du sift.
    Ensuite de calculer l'image résultante de la différence de Gauss (DoG)
    
    paramètre:img, c'est le chemin de l'image sur laquelle on souhaite travailler   
    """    
    
    img1= cv2.imread(img)                       # lecture image
    img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) # conversion en niveau de gris
    img1=cv2.GaussianBlur(img1,(3,3),0)         # calcul du gradient sur limage avec le nombre voulu et et coef du filtre 
    
    img2=cv2.GaussianBlur(img1,(3,3),1.6)
    img3=cv2.GaussianBlur(img1,(3,3),0.4)
    
    
    
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    rows3 = img3.shape[0]
    cols3 = img3.shape[1]
    
      # sortie qui contient les images fusionnées dans même fenêtre
    out = np.zeros((max([rows1,rows2,rows3]),cols1+cols2+cols3,3), dtype='uint8')
    
        # posionnement de la 1ere image a gauche
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])
    
        # posionnement de la 2eme image au milieu droite
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])
    
    # posionnement de la 2eme image a droite    
    out[:rows3,cols1+cols2:cols1+cols2+cols3,:] = np.dstack([img3, img3, img3])
    cv2.imwrite("blur.jpg",out)
    
    #difference de gaussienne DoG
    
    img4=img2-img1
    img5=img2-img3
    rows4 = img4.shape[0]
    cols4 = img4.shape[1]
    rows5 = img5.shape[0]
    cols5 = img5.shape[1]
    out1 = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    
        # posionnement de la 1ere image a gauche
    out1[:rows1,:cols1,:] = np.dstack([img4, img4, img4])
    
        # posionnement de la 2eme image a droite
    out1[:rows2,cols1:cols1+cols2,:] = np.dstack([img5, img5, img5])
    
    cv2.imwrite("DoG.jpg",out1)
    cv2.namedWindow("main", cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow("main", out1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()