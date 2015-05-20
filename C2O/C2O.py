# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:08:26 2015

@author: etienne
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def diff(img,dX,dY):
    """
    Fonction de calcul de la difference d'image par rapport a une version decalee d'elle meme.
    Cette fonction est apellee par la fonction de calcul du descripteur C2O, le parametre dX et dY 
    y sont egalement calcules en fonction de l'orientation et de la norme du vecteur de deplacement.
    
    
    """
    
    rows,cols , plans = img.shape
    
    M = np.float32([[1,0,-dX],[0,1,dY]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    print rows , cols
    

    
    reMatdst=np.zeros([rows-dY,cols-dX,plans])
    reMatdst=dst[dY:rows,0:cols-dX,:]
    
    
    reMatimg=np.zeros([rows-dY,cols-dX,plans])
    reMatimg=img[dY:rows,0:cols-dX,:]
    
    rows,cols, plans = reMatdst.shape
    
    print rows , cols    
    
    cv2.imshow('image decalée',dst)
    cv2.waitKey(0) 
    
    cv2.imshow('image decalée redimensionnee',reMatdst)
    cv2.waitKey(0)    
    
    cv2.imshow('image d origine redimensionnée',reMatimg)
    cv2.waitKey(0)
 

    return reMatimg-reMatdst
    
    
    



def RGBtoLAB(imag):
    
#    imag = cv2.imread(img,1)
    rows,cols, plans = imag.shape

 
    imag=imag.astype(np.float32)
    
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
    
    test = XBIN-1    
    

    print "min XBIN"
    print np.min(XBIN)
    print "min XBIN-1"
    print np.min(XBIN-1)    
    
    #L
    Lab[:,:,0] = (116*( (XYZ[:,:,1])*YBIN)**(1.0/3.0))-16 -903.3*( (XYZ[:,:,1])* (YBIN-1))
    #A
    Lab[:,:,1] = 500*((XBIN*(XYZ[:,:,0])**(1.0/3.0))-(YBIN*(XYZ[:,:,1])**(1.0/3.0))+((YBIN-1)*(XYZ[:,:,1])*(7.787)+(1.0/16.0))-((XBIN-1)*(XYZ[:,:,0])*(7.787)+(1.0/16.0)))
    #B
    Lab[:,:,2] = 300*((YBIN*(XYZ[:,:,1])**(1.0/3.0))-(ZBIN*(XYZ[:,:,2])**(1.0/3.0))+((ZBIN-1)*(XYZ[:,:,2])*(7.787)+(1.0/16.0))-((YBIN-1)*(XYZ[:,:,1])*(7.787)+(1.0/16.0)))
    print np.max(Lab[:,:,1]) , np.max(Lab[:,:,2])

    

    return Lab
    
    

def CarthesianToSpheric(Lab):
    
    rows,cols, plans = Lab.shape
    Spheric=np.zeros((rows,cols,plans))
    
    Spheric[:,:,0] = np.sqrt((Lab[:,:,0]**2)+(Lab[:,:,1]**2)+(Lab[:,:,2]**2))
#    Spheric[:,:,1] = np.arcsin(Lab[:,:,1]/np.sqrt((Lab[:,:,0]**2)+(Lab[:,:,1]**2)))
    Spheric[:,:,1] = np.arctan2(Lab[:,:,1],Lab[:,:,0])
#    Spheric[:,:,2] = np.arccos(Lab[:,:,2]/Spheric[:,:,0])

    Spheric[:,:,2] = np.arctan2(np.sqrt((Lab[:,:,0]**2)+(Lab[:,:,1]**2)),Spheric[:,:,0])
    
    return Spheric




def SphericToCartesian(Spheric):
    rows,cols, plans = Spheric.shape
    Cartesian=np.zeros((rows,cols,plans))
    
    Cartesian[:,:,0] = Spheric[:,:,0] * np.sin(Spheric[:,:,2]) *  np.cos(Spheric[:,:,1])
    Cartesian[:,:,1] = Spheric[:,:,0] * np.sin(Spheric[:,:,2]) *  np.sin(Spheric[:,:,1])
    Cartesian[:,:,2] = Spheric[:,:,0] * np.cos(Spheric[:,:,2])
    
    
        
    
    return Cartesian
 






#test = np.zeros((1,1,3))
#test[0,0,0] = 1
#test[0,0,1] = 1
#test[0,0,2] = 1
#print "Coordonnees d origine"
#print test
#
#teste=CarthesianToSpheric(test)
#print "Coordonnees spheriques"
#print teste
#
#test2 = SphericToCartesian(teste)
#
#print "Coordonnees reconverties"
#print test2

V = 127.0

imgtest = np.zeros((100,300,3))
imgtest[0:100,0:30]= [[0.0,V,0.0]]
imgtest[0:100,30:60]= [[0.0,0.0,V]]
imgtest[0:100,60:90]= [[0.0,V,0.0]]
imgtest[0:100,90:120]= [[0.0,0.0,V]]
imgtest[0:100,120:150]= [[0.0,V,0.0]]
imgtest[0:100,150:180]= [[0.0,0.0,V]]
imgtest[0:100,180:210]= [[0.0,V,0.0]]
imgtest[0:100,210:240]= [[0.0,0.0,V]]
imgtest[0:100,240:270]= [[0.0,V,0.0]]
imgtest[0:100,270:300]= [[0.0,0.0,V]]


#imag = cv2.imread("Food.0006.ppm",1)
#imag = cv2.imread("test2.jpg",1)
cv2.imshow('imgtest',imgtest)
cv2.waitKey(0)
imgtest=imgtest.astype(np.float32)


#Lab=RGBtoLAB("Food.0006.ppm")
#Lab=RGBtoLAB(imgtest)



#imag = cv2.imread("Food.0006.ppm",1)
#imag = cv2.imread("test2.jpg",1)
#rows,cols, plans = imag.shape
#
#Lab=np.zeros((rows,cols,3))
#
#cv2.imshow('image d origine',imag)
#cv2.waitKey(0)
#imag=imag.astype(np.float32)
##imag*=(1.0/255.0)
Lab=cv2.cvtColor(imgtest, cv2.COLOR_BGR2LAB) 
##
#print "Apres la conversion LAB"
#print np.max(Lab[:,:,0]) , np.max(Lab[:,:,1]) , np.max(Lab[:,:,2])


#cv2.imshow('L',Lab[:,:,0]/np.max(Lab[:,:,0]))
#cv2.waitKey(0)
#cv2.imshow('a',Lab[:,:,1]/np.max(Lab[:,:,1]))
#cv2.waitKey(0)
#cv2.imshow('b',Lab[:,:,2]/np.max(Lab[:,:,2]))
#cv2.waitKey(0)


cv2.imshow('L',Lab[:,:,0])
cv2.waitKey(0)
cv2.imshow('a',Lab[:,:,1])
cv2.waitKey(0)
cv2.imshow('b',Lab[:,:,2])
cv2.waitKey(0)
LabDiff=0
LabDiff = diff(Lab,1,0)

print "apres la diff"
print np.max(Lab[:,:,0]) , np.max(Lab[:,:,1]) , np.max(Lab[:,:,2]) , LabDiff.shape

cv2.imshow('Image de la difference',LabDiff[:,:,0])
cv2.waitKey(0)

SphereCoord=CarthesianToSpheric(LabDiff)


print "Max des coordonnes spheriques"
print np.max(SphereCoord[:,:,0]) , np.max(SphereCoord[:,:,1]) , np.max(SphereCoord[:,:,2])

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

L=Lab[:,:,0]
a=Lab[:,:,1]
b=Lab[:,:,2]


ax.contourf(LabDiff[:,:,1], LabDiff[:,:,2], LabDiff[:,:,0], zdir='z',offset=-190)
ax.contourf(LabDiff[:,:,1], LabDiff[:,:,2], LabDiff[:,:,0], zdir='y',offset=-190)
ax.contourf(LabDiff[:,:,1], LabDiff[:,:,2], LabDiff[:,:,0], zdir='x',offset=-190)
ax.scatter3D(LabDiff[:,:,1], LabDiff[:,:,2], LabDiff[:,:,0])



cv2.destroyAllWindows()
ax.set_xlabel('a')
ax.set_ylabel('b')
ax.set_zlabel('L')