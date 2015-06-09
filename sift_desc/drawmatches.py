# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 23:01:55 2015

@author: Khalil
"""

import function_sift as fc
import numpy as np
import cv2

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    Cette fonction est l'equivalent de cv2.drawMatches de OpenCV pour tester notre descripteur sift 

    Elle permet d'associer les points clés qui se ressemblent les plus entre deux images 


    img1,img2 - les images en niveaux de gris
    kp1,kp2 - Les points clés détectectés dans les deux images 
                en utilisant la méthode sift dans notre cas
                
    matches - Liste de correspondance entre les points cles des images
    """

    # Creation d'une image qui est composee des deux images mises cote a cote
    
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # posionnement de la 1ere image a gauche
    out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

    # posionnement de la 2eme image a droite
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

    
    for mat in matches:

        # Les points cles correspondant a chaque image
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # coordonnees x et y des points
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # parametres des cercles pour la represntation des points cles 
        # rayon= 4 couleur blue epaisseur = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # tracer de la droite entre deux points cles
        # couleur blue epaisseur = 1
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


    # Affichage du resltat 
    cv2.imshow('Matched Features', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Application de la fonction 
#chemins des images
fold1='C:\\Users\RTMA\\Desktop\\Nouveau dossier\\khalil\\Nouveau dossier\\662.jpg'
fold2='C:\\Users\RTMA\\Desktop\\Nouveau dossier\\khalil\\Nouveau dossier\\662_r.jpg'

# detection des points cles image 1
kp1,des1 = fc.SIFT(fold1)

# detection des points cles image 2
kp2,des2 = fc.SIFT(fold2)

#lecture des images 
img1 = cv2.imread(fold1) 
img2 = cv2.imread(fold2)

#conversion en niveau de gris
img1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# faire la correspondance
des1=np.uint8(des1) 
des2=np.uint8(des2)
matches = bf.match(des1,des2)


matches = sorted(matches, key=lambda val: val.distance)

drawMatches(img1, kp1, img2, kp2, matches[:20])