# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:08:26 2015

@author: etienne
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def diff(imga,dX,dY):
    """
    Fonction de calcul de la difference d'image par rapport a une version decalee d'elle meme.
    Cette fonction est apellee par la fonction de calcul du descripteur C2O, le parametre dX et dY 
    y sont egalement calcules en fonction de l'orientation et de la norme du vecteur de deplacement.
    
    
    """
    
    img = cv2.imread(imga,0)
    rows,cols = img.shape
    
    M = np.float32([[1,0,-dX],[0,1,dY]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    print rows , cols
    

    
    reMatdst=np.zeros([rows-dY,cols-dX])
    reMatdst=dst[dY:rows,0:cols-dX]
    
    
    reMatimg=np.zeros([rows-dY,cols-dX])
    reMatimg=img[dY:rows,0:cols-dX]
    
    rows,cols = reMatdst.shape
    
    print rows , cols    
    
#    cv2.imshow('img',dst)
#    cv2.waitKey(0) 
#    
#    cv2.imshow('img',reMatimg)
#    cv2.waitKey(0)
#    
#    cv2.imshow('img',reMatdst)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()   

    return reMatimg-reMatdst
    
    
    



def RGBtoLAB(img):
    
    imag = cv2.imread(img,1)
    rows,cols, plans = imag.shape

 
    
    
    XYZ=np.zeros((rows,cols,3))
    XBIN=np.zeros((rows,cols))
    YBIN=np.zeros((rows,cols))
    ZBIN=np.zeros((rows,cols))
    Lab=np.zeros((rows,cols,3))

    
    XYZ[:,:,0] = imag[:,:,2]*0.618 + imag[:,:,1]*0.177 + imag[:,:,0]*0.205
    XYZ[:,:,1] = imag[:,:,2]*0.299 + imag[:,:,1]*0.587 + imag[:,:,0]*0.114
    XYZ[:,:,2] = imag[:,:,2]*0 + imag[:,:,1]*0.056 + imag[:,:,0]*0.944
    
    XYZ= XYZ/255.0
    
    
    
    XBIN[XYZ[:,:,0]>0.008856]=1
    YBIN[XYZ[:,:,1]>0.008856]=1
    ZBIN[XYZ[:,:,2]>0.008856]=1
    

    
    Lab[:,:,0] = (116*( (XYZ[:,:,1])*YBIN)**(1.0/3.0))-16 -903.3*( (XYZ[:,:,1])* (YBIN-1))
    Lab[:,:,1] = 500*((XBIN*(XYZ[:,:,0])**(1.0/3.0))-(YBIN*(XYZ[:,:,1])**(1.0/3.0))+((YBIN-1)*(XYZ[:,:,1])*(7.787)+(1.0/16.0))-((XBIN-1)*(XYZ[:,:,0])*(7.787)+(1.0/16.0)))
    Lab[:,:,2] = 300*((YBIN*(XYZ[:,:,1])**(1.0/3.0))-(ZBIN*(XYZ[:,:,2])**(1.0/3.0))+((ZBIN-1)*(XYZ[:,:,2])*(7.787)+(1.0/16.0))-((YBIN-1)*(XYZ[:,:,1])*(7.787)+(1.0/16.0)))
#    Lab[:,:,0] = 116*( (XYZ[:,:,1])*YBIN)**(1/3)-16
    print np.max(Lab[:,:,1]) , np.max(Lab[:,:,2])

#    cv2.imshow('img',Lab[:,:,0]/np.max(Lab[:,:,0]))
#    cv2.waitKey(0)
#    cv2.imshow('img',Lab[:,:,1]/np.max(Lab[:,:,1]))
#    cv2.waitKey(0)
#    cv2.imshow('img',Lab[:,:,2]/np.max(Lab[:,:,2]))
#    cv2.waitKey(0)
#    cv2.destroyAllWindows() 
    
    

    return Lab
    
    

def CarthesianToSpheric(Lab):
    
    rows,cols, plans = Lab.shape
    Spheric=np.zeros((rows,cols,plans))
    
    Spheric[:,:,0] = np.sqrt(Lab[:,:,0]**2+Lab[:,:,1]**2+Lab[:,:,2]**2)
    Spheric[:,:,1] = np.arctan(Lab[:,:,1]/Lab[:,:,0])
    Spheric[:,:,2] = np.arctan(Lab[:,:,0]/Spheric[:,:,0])
    
    return Spheric

test=RGBtoLAB("20.jpg")

teste=CarthesianToSpheric(test)

#Axes3D.scatter(teste[:,:,0], teste[:,:,1], teste[:,:,2], zdir=u'z', s=20, c=u'b', depthshade=True)

print np.max(teste[:,:,2]) , np.min(teste[:,:,2])

#fig = plt.figure()
#
#ax = fig.add_subplot(111, projection='3d')
#
#ax.scatter(teste[:,:,0], teste[:,:,1], teste[:,:,2])