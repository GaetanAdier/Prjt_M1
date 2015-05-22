# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:08:26 2015

@author: etienne
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 



#######################################################################################################################
#####################################################Fonctions#########################################################
#######################################################################################################################

def diff(img,dX,dY):
    """
    Fonction de calcul de la difference d'image par rapport a une version decalee d'elle meme.
    Cette fonction est apellee par la fonction de calcul du descripteur C2O, le parametre dX et dY 
    y sont egalement calcules en fonction de l'orientation et de la norme du vecteur de deplacement.
    
    
    """
    
    rows,cols , plans = img.shape
    
    M = np.float32([[1,0,-dX],[0,1,dY]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    

    
    reMatdst=np.zeros([rows-dY,cols-dX,plans])
    reMatdst=dst[dY:rows,0:cols-dX,:]
    
    
    reMatimg=np.zeros([rows-dY,cols-dX,plans])
    reMatimg=img[dY:rows,0:cols-dX,:]
    
    Res = np.zeros([rows-dY,cols-dX,plans])
    Res[:,:,0]=reMatimg[:,:,0]-reMatdst[:,:,0]
    Res[:,:,1]=reMatimg[:,:,1]-reMatdst[:,:,1]
    Res[:,:,2]=reMatimg[:,:,2]-reMatdst[:,:,2]
 

    return Res
    
    
    



def RGBtoLAB(imag):
    
    """
    Fonction de BGR à l'espace L*a*b*. Cette fonction utilise le passage par l'espace XYZ. La matrice de passage est speifiee en interne
    
    TODO : La fonction n'est pas utilisee pour le moment, le matrice de passage devra être mise en parametre de la fonction ainsi que le blanc de reference utilise
    """
    
    rows,cols, plans = imag.shape

 
    imag=imag.astype(np.float32)
    
    XYZ=np.zeros((rows,cols,3))
    XBIN=np.zeros((rows,cols))
    YBIN=np.zeros((rows,cols))
    ZBIN=np.zeros((rows,cols))
    Lab=np.zeros((rows,cols,3))
    MatPass = np.zeros((3,3))
    
#    0.4887180  0.3106803  0.2006017
#    0.1762044  0.8129847  0.0108109
#    0.0000000  0.0102048  0.9897952
    #Matrice de passage RGB->XYZ
    MatPass[0,0]=  0.4887180
    MatPass[0,1]= 0.3106803
    MatPass[0,2]= 0.2006017
    MatPass[1,0]= 0.1762044
    MatPass[1,1]=0.8129847
    MatPass[1,2]= 0.0108109
    MatPass[2,0]= 0
    MatPass[2,1]= 0.0102048
    MatPass[2,2]=0.9897952

    # Blanc de reference
    stdIllum = np.zeros(3)  
    stdIllum[0]= 255.0  
    stdIllum[1]= 255.0
    stdIllum[2]= 255.0
    
    XYZ[:,:,0] = (imag[:,:,2]*MatPass[0,2] + imag[:,:,1]*MatPass[0,1] + imag[:,:,0]*MatPass[0,0])/stdIllum[0]
    XYZ[:,:,1] = (imag[:,:,2]*MatPass[1,2] + imag[:,:,1]*MatPass[1,1] + imag[:,:,0]*MatPass[1,0])/stdIllum[1]
    XYZ[:,:,2] = (imag[:,:,2]*MatPass[2,2] + imag[:,:,1]*MatPass[2,1] + imag[:,:,0]*MatPass[2,0])/stdIllum[2]
    

    
#    XYZ= XYZ/255.0
    
    
    
    XBIN[XYZ[:,:,0]>0.008856]=1
    YBIN[XYZ[:,:,1]>0.008856]=1
    ZBIN[XYZ[:,:,2]>0.008856]=1
    
    test = XBIN-1    
    

#    print "min XBIN"
#    print np.min(XBIN)
#    print "min XBIN-1"
#    print np.min(XBIN-1)    
    
    #L
    Lab[:,:,0] = (116*( (XYZ[:,:,1])*YBIN)**(1.0/3.0))-16 -903.3*( (XYZ[:,:,1])* (YBIN-1))
    #A
    Lab[:,:,1] = 500*((XBIN*(XYZ[:,:,0])**(1.0/3.0))-(YBIN*(XYZ[:,:,1])**(1.0/3.0))+((YBIN-1)*(XYZ[:,:,1])*(7.787)+(1.0/16.0))-((XBIN-1)*(XYZ[:,:,0])*(7.787)+(1.0/16.0)))
    #B
    Lab[:,:,2] = 300*((YBIN*(XYZ[:,:,1])**(1.0/3.0))-(ZBIN*(XYZ[:,:,2])**(1.0/3.0))+((ZBIN-1)*(XYZ[:,:,2])*(7.787)+(1.0/16.0))-((YBIN-1)*(XYZ[:,:,1])*(7.787)+(1.0/16.0)))
#    print np.max(Lab[:,:,1]) , np.max(Lab[:,:,2])

    

    return Lab
    
    

def CarthesianToSpheric(Lab):
    """
    Fonction de calcul des coordonnees spheriques a partir des coordonnes carthesiennes
    
    """
    
    rows,cols, plans = Lab.shape
    Spheric=np.zeros((rows,cols,plans))
    
    Spheric[:,:,0] = np.sqrt((Lab[:,:,0]**2)+(Lab[:,:,1]**2)+(Lab[:,:,2]**2))
#    Spheric[:,:,1] = np.arcsin(Lab[:,:,1]/np.sqrt((Lab[:,:,0]**2)+(Lab[:,:,1]**2)))
    Spheric[:,:,1] = np.arctan2(Lab[:,:,1],Lab[:,:,0])
#    Spheric[:,:,2] = np.arccos(Lab[:,:,2]/Spheric[:,:,0])

    Spheric[:,:,2] = np.arctan2(np.sqrt((Lab[:,:,0]**2)+(Lab[:,:,1]**2)),Spheric[:,:,0])
    
    return Spheric




def SphericToCartesian(Spheric):
    """
    Fonction de calcul des coordonnees carthesiennes a partir des coordonnes spheriques
    
    """
    rows,cols, plans = Spheric.shape
    Cartesian=np.zeros((rows,cols,plans))
    
    Cartesian[:,:,0] = Spheric[:,:,0] * np.sin(Spheric[:,:,2]) *  np.cos(Spheric[:,:,1])
    Cartesian[:,:,1] = Spheric[:,:,0] * np.sin(Spheric[:,:,2]) *  np.sin(Spheric[:,:,1])
    Cartesian[:,:,2] = Spheric[:,:,0] * np.cos(Spheric[:,:,2])
    
    
    return Cartesian
 




#######################################################################################################################
#######################################################Main############################################################
#######################################################################################################################

#################################
## Definition de l'image de test#
#################################
V =100
#BGR
#Test Bleu Jaune
#mat2 = [[0.0,V,V]]
#mat1= [[V,0.0,0.0]]

#Test Rouge Vert
mat1 = [[0.0,V,0.0]]
mat2= [[0.0,0.0,V]]

imgtest = np.zeros((100,300,3))
imgtest[0:100,0:30]= mat1
imgtest[0:100,30:60]= mat2
imgtest[0:100,60:90]= mat1
imgtest[0:100,90:120]= mat2
imgtest[0:100,120:150]= mat1
imgtest[0:100,150:180]= mat2
imgtest[0:100,180:210]= mat1
imgtest[0:100,210:240]= mat2
imgtest[0:100,240:270]= mat1
imgtest[0:100,270:300]= mat2


##################################################################
###Ouverture de l'image et conrversion en 32bit#######
##################################################################

#imag = cv2.imread("Food.0006.ppm",1)
#imag = cv2.imread("test2.jpg",1)
cv2.imshow('imgtest',imgtest)
cv2.waitKey(0)
imgtest=imgtest.astype(np.float32)
imag=imag.astype(np.float32)

#Lab=RGBtoLAB(imgtest)
#Lab=RGBtoLAB(imag)

##################################################################
#######Passage de l'image dans l'espace Lab#######################
##################################################################

Lab=cv2.cvtColor(imgtest, cv2.COLOR_BGR2LAB) 
#Lab=cv2.cvtColor(imag, cv2.COLOR_BGR2LAB) 

#print "L"
#print Lab[:,:,0]
###
#print "Apres la conversion LAB"
#print np.max(Lab[:,:,0]) , np.max(Lab[:,:,1]) , np.max(Lab[:,:,2])





#cv2.imshow('L',Lab[:,:,0])
#cv2.waitKey(0)
#cv2.imshow('a',Lab[:,:,1])
#cv2.waitKey(0)
#cv2.imshow('b',Lab[:,:,2])
#cv2.waitKey(0)

##################################################################
##### Fonction de calcul de la difference#########################
##################################################################
LabDiff=0
LabDiff = diff(Lab,1,0)

#print "apres la diff"
#print np.max(LabDiff[:,:,0]) , np.max(LabDiff[:,:,1]) , np.max(LabDiff[:,:,2]) , LabDiff.shape
#
#cv2.imshow('Image de la difference (luminance)',LabDiff[:,:,0])
#cv2.waitKey(0)

#SphereCoord=CarthesianToSpheric(LabDiff)
#
#
#print "Max des coordonnes spheriques"
#print np.max(SphereCoord[:,:,0]) , np.max(SphereCoord[:,:,1]) , np.max(SphereCoord[:,:,2])


L=Lab[:,:,0]
a=Lab[:,:,1]
b=Lab[:,:,2]



#######################################################################################################################
#####################################################Affichage#########################################################
#######################################################################################################################
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')




extends = ["neither", "both", "min", "max"]
ax.scatter3D(LabDiff[:,:,1], LabDiff[:,:,2], LabDiff[:,:,0])
ax.set_xlim([-200, 200])
ax.set_ylim([-200, 200])
ax.set_zlim([-80, 80])
ax.contourf(LabDiff[:,:,1], LabDiff[:,:,2], LabDiff[:,:,0], zdir='z',offset=-80)
ax.contourf(LabDiff[:,:,1], LabDiff[:,:,2], LabDiff[:,:,0], zdir='y',offset=-200)
ax.contourf(LabDiff[:,:,1], LabDiff[:,:,2], LabDiff[:,:,0], zdir='x',offset=-200)
#cv2.destroyAllWindows()
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('L')

plt.show()
