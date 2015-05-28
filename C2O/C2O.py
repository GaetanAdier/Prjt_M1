# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:08:26 2015

@author: etienne
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from multiprocessing import Process, Queue
import time  



#######################################################################################################################
####################################################Class timer########################################################
#######################################################################################################################
class Timer(object):  
    def start(self):  
        if hasattr(self, 'interval'):  
            del self.interval  
        self.start_time = time.time()  
  
    def stop(self):  
        if hasattr(self, 'start_time'):  
            self.interval = time.time() - self.start_time  
            del self.start_time # Force timer reinit  


#######################################################################################################################
#####################################################Fonctions#########################################################
#######################################################################################################################

def diff(img,dX,dY):
    """
    Fonction de calcul de la difference d'image par rapport a une version decalee d'elle meme.
    Cette fonction est apellee par la fonction de calcul du descripteur C2O, le parametre dX et dY 
    y sont egalement calcules en fonction de l'orientation et de la norme du vecteur de deplacement.
    
    :param img: A matrix containing the image on which you need the C2O feature calculation.
    :type img: np.ndarray    
    :param dX: Number of pixel to shift on the X axis.
    :type dX: float   
    :param dY: Number of pixel to shift on the X axis.
    :type dY: float 
    
    """
    
        
    
    rows,cols , plans = img.shape
    
    #Image shifting
    M = np.float32([[1,0,-dX],[0,1,dY]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    NegY=0 
    NegX=0
    

    
#    reMatdst=np.zeros([rows-dY,cols-dX,plans])
#    reMatdst=dst[dY:rows,0:cols-dX,:]
#    
#    
#    reMatimg=np.zeros([rows-dY,cols-dX,plans])
#    reMatimg=img[dY:rows,0:cols-dX,:]
    
    # Gestion of negative shifting
    if dY<0:
        NegY=-dY
    else:
        NegY=0
        
    if dX<0:
        NegX=-dX
    else:
        NegX=0
    
    dX=np.abs(dX)
    dY=np.abs(dY)
    
    #Image resizing
    reMatdst=np.zeros([rows-dY,cols-dX,plans])
    reMatdst=dst[dY-NegY:rows-NegY,NegX:cols-dX+NegX,:]
    
    
    reMatimg=np.zeros([rows-dY,cols-dX,plans])
    reMatimg=img[dY-NegY:rows-NegY,NegX:cols-dX+NegX,:]    
    
    
    Res = np.zeros([rows-dY,cols-dX,plans])

    
    Res = reMatimg-reMatdst
    
    # Uncomment to display the coocurence matrix
    
#    fig = plt.figure()
#
#    ax = fig.add_subplot(111, projection='3d')
#    
#    ax.scatter3D(Res[:,:,1], Res[:,:,2], Res[:,:,0],zdir='z')
#    ax.set_xlim([-60, 60])
#    ax.set_ylim([-60, 60])
#    ax.set_zlim([-80, 80])
#    ##ax.contourf(Res[:,:,1], Res[:,:,2], Res[:,:,0], zdir='z',offset=np.min(LabDiff[:,:,0]), cmap='coolwarm')
#    ##ax.contourf(Res[:,:,1], Res[:,:,2], Res[:,:,0], zdir='y',offset=np.min(LabDiff[:,:,2]), cmap='coolwarm')
#    ##ax.contourf(Res[:,:,1], Res[:,:,2], Res[:,:,0], zdir='x',offset=-np.min(LabDiff[:,:,1]), cmap='coolwarm')
#    #cv2.destroyAllWindows()
#    ax.set_xlabel('a')
#    ax.set_ylabel('b')
#    ax.set_zlabel('L')
 

    return Res
    
    
    



def RGBtoLAB(imag, MatPass, stdIllum):
    
    """
    Fonction de BGR à l'espace L*a*b*. Cette fonction utilise le passage par l'espace XYZ. La matrice de passage est speifiee en interne
    
    :param imag: A matrix containing the image on which you need the C2O feature calculation.
    :type imag: np.ndarray
    :param MatPass: The transition matrix for the BGR to XYZ transformation
    :type MatPass: np.ndarray
    :param stdIllum: The standard illuminant choosen for the BGR to XYZ transformation.
    :type stdIllum: np.ndarray
    
    :return: The coocurence matrix in spherical coordinates.
    :rtype: np.ndarray    
    
    """
    
    rows,cols, plans = imag.shape

 
    imag=imag.astype(np.float32)
    
    # Set values in the range [0,1]
    imag = imag/np.max(imag)
    
    XYZ=np.zeros((rows,cols,3))
    XBIN=np.zeros((rows,cols))
    YBIN=np.zeros((rows,cols))
    ZBIN=np.zeros((rows,cols))
    Lab=np.zeros((rows,cols,3))
    
    


    # Transformation throught the XYZ space
    XYZ[:,:,0] = (imag[:,:,2]*MatPass[0,0] + imag[:,:,1]*MatPass[0,1] + imag[:,:,0]*MatPass[0,2])/stdIllum[0]
    XYZ[:,:,1] = (imag[:,:,2]*MatPass[1,0] + imag[:,:,1]*MatPass[1,1] + imag[:,:,0]*MatPass[1,2])/stdIllum[1]
    XYZ[:,:,2] = (imag[:,:,2]*MatPass[2,0] + imag[:,:,1]*MatPass[2,1] + imag[:,:,0]*MatPass[2,2])/stdIllum[2]
    

    
    # Thresholding matrix
    XBIN[XYZ[:,:,0]>0.008856]=1
    YBIN[XYZ[:,:,1]>0.008856]=1
    ZBIN[XYZ[:,:,2]>0.008856]=1
     
    
   
    
    #L
    Lab[:,:,0] = (116*( (XYZ[:,:,1])*YBIN)**(1.0/3.0))-16 -903.3*( (XYZ[:,:,1])* (YBIN-1))
    #A
    Lab[:,:,1] = 500*((XBIN*(XYZ[:,:,0])**(1.0/3.0))-(YBIN*(XYZ[:,:,1])**(1.0/3.0))+((YBIN-1)*(XYZ[:,:,1])*(7.787)+(16.0/116.0))-((XBIN-1)*(XYZ[:,:,0])*(7.787)+(16.0/116.0)))
    #B
    Lab[:,:,2] = 200*((YBIN*(XYZ[:,:,1])**(1.0/3.0))-(ZBIN*(XYZ[:,:,2])**(1.0/3.0))+((ZBIN-1)*(XYZ[:,:,2])*(7.787)+(16.0/116.0))-((YBIN-1)*(XYZ[:,:,1])*(7.787)+(16.0/116.0)))


    

    return Lab
    
    

def CarthesianToSpheric(Lab):
    """
    Function for computing spherical coordinates from carthesian ones.
    
    :param Lab: The coocurence matrix in carthesian coordinates.
    :type Lab: np.ndarray
    
    :return: The coocurence matrix in spherical coordinates.
    :rtype: np.ndarray
    
    """
    
    rows,cols, plans = Lab.shape
    Spheric=np.zeros((rows,cols,plans))
    
    Spheric[:,:,0] = np.sqrt((Lab[:,:,0]**2)+(Lab[:,:,1]**2)+(Lab[:,:,2]**2))
    Spheric[:,:,1] = np.arctan2(Lab[:,:,1],Lab[:,:,0])
    Spheric[:,:,2] = np.arctan2(np.sqrt((Lab[:,:,0]**2)+(Lab[:,:,1]**2)),Lab[:,:,2])
    
    # Remise a l'echelle des valeurs d'angles 
    Spheric[:,:,1][Spheric[:,:,1]<0]=Spheric[:,:,1][Spheric[:,:,1]<0]+(2*np.pi)
    Spheric[:,:,2]=  Spheric[:,:,2] - np.pi/2
    return Spheric




def SphericToCartesian(Spheric):
    """
    Function for computing carthesian coordinates from spherical ones.
    
    :param Spheric: The coocurence matrix in spherical coordinates.
    :type Spheric: np.ndarray
    
    """
    rows,cols, plans = Spheric.shape
    Cartesian=np.zeros((rows,cols,plans))
    
    Cartesian[:,:,0] = Spheric[:,:,0] * np.sin(Spheric[:,:,2]) *  np.cos(Spheric[:,:,1])
    Cartesian[:,:,1] = Spheric[:,:,0] * np.sin(Spheric[:,:,2]) *  np.sin(Spheric[:,:,1])
    Cartesian[:,:,2] = Spheric[:,:,0] * np.cos(Spheric[:,:,2])
    
    
    return Cartesian
 


def SphericQuantif(C2OMat, NE, Nalpha, Nbeta):
    """
    
    Function for computing the spherical quantization for the C2O signature.
    
    :param C2OMat: The coocurence matrix in spherical coordinates.
    :type C2OMat: np.ndarray
    :param Nalpha: Number of intervals considered for the signature calculation on the \alpha component.
    :type Nalpha: float
    :param Nbeta: Number of intervals considered for the signature calculation on the \beta component.
    :type Nbeta: float
    

    
    :return: The signature of the image 
    :rtype: A vector of 4*Nalpha*Nbeta float64
    """

    rows,cols, plans = C2OMat.shape

    BinE=np.zeros((rows,cols))
    BinAlpha=np.zeros((rows,cols))
    BinBeta=np.zeros((rows,cols))
    SigC2O = np.zeros(Nalpha*Nbeta*NE)
    n=0

    
    
    for i in np.arange(0,NE):
        BinE=np.zeros((rows,cols))
        BinE[(C2OMat[:,:,0]>=(9.0/(NE-1))*i)&(C2OMat[:,:,0]<=(9.0/(NE-1))*(i+1))]=1
        if i > NE-2 :
            BinE[(C2OMat[:,:,0]>=(9.0/(NE-1))*i)]=1
        for j in np.arange(0,Nbeta):
            BinBeta=np.zeros((rows,cols))
            BinBeta[(BinE==1)&(C2OMat[:,:,2]>=((-np.pi/2)+((np.pi/Nbeta)*j)))&(C2OMat[:,:,2]<=((-np.pi/2)+((np.pi/Nbeta)*(j+1))))]=1
            for k in np.arange(0,Nalpha):
                BinAlpha=np.zeros((rows,cols))
                BinAlpha[(BinBeta==1)&(C2OMat[:,:,1]>=(((2.0*np.pi/Nalpha)*k)-(np.pi/(2*Nalpha))))&(C2OMat[:,:,1]<=(((2.0*np.pi/Nalpha)*(k+1))-(np.pi/(2*Nalpha))))]=1
                
                

                SigC2O[n]=np.sum(BinAlpha)
                
                n = n+1
                
    return SigC2O
    
    
    
def C2O(image, NormLambda, RadLambda, NE, Nalpha, Nbeta, SigC2Ot):
    """
    Function for the C2O feature calculation
    
    :param image: Path directory of the image on which you need the C2O feature calculation.
    :type image: string
    :param NormLambda: Norm of \Delta vector for the image color diference.
    :type NormLambda: float
    :param RadLambda: Radix of \Delta vector for the image color diference.
    :type RadLambda: float
    :param Nalpha: Number of intervals considered for the signature calculation on the \alpha component.
    :type Nalpha: float
    :param Nbeta: Number of intervals considered for the signature calculation on the \beta component.
    :type Nbeta: float
    :param SigC2Ot: Return parameter for the parralelise version.
    :type SigC2Ot: queue
    
    :return: The signature of the image in the RadLambda orientation and at the NormLambda distance for color difference
    :rtype: A vector of NE*Nalpha*Nbeta float64
    
    """
    dX = 0
    dY = 0    
    
    #Matrice de passage RGB->XYZ (OpenCV)
    MatPass[0,0]=  0.412453
    MatPass[0,1]= 0.357580
    MatPass[0,2]= 0.180423
    MatPass[1,0]= 0.212671
    MatPass[1,1]=0.715160
    MatPass[1,2]= 0.072169
    MatPass[2,0]= 0.019334
    MatPass[2,1]= 0.119193
    MatPass[2,2]=0.950227
    
    # Blanc de reference (F7)
    stdIllum = np.zeros(3)  
    stdIllum[0]= 0.95041  
    stdIllum[1]= 1.00000
    stdIllum[2]= 1.08747
    
    imag = cv2.imread(image,1)
    
    #Transformation throught the Lab space
    Lab=RGBtoLAB(imag,MatPass, stdIllum)
    
    LabDiff=0
    # Calculation of the shifting parameter
    dX = np.round(np.cos(RadLambda)*NormLambda)
    dY = np.round(np.sin(RadLambda)*NormLambda)
    

    
    # Calculation of the difference of color
    LabDiff = diff(Lab,dX,dY)
    
    # Transformation in spherical coordinates
    SphereCoord=CarthesianToSpheric(LabDiff)


    # Calculation of the signature by the spherical quantization
    SigC2O = SphericQuantif(SphereCoord,NE ,Nalpha,Nbeta)
    
    # Assignation of te result on the queue for the parralelized version
    SigC2Ot.put(SigC2O)
    
    return SigC2O


#######################################################################################################################
#######################################################Main############################################################
#######################################################################################################################

MatPass = np.zeros((3,3))
#Matrice de passage RGB->XYZ (OpenCV)
MatPass[0,0]=  0.412453
MatPass[0,1]= 0.357580
MatPass[0,2]= 0.180423
MatPass[1,0]= 0.212671
MatPass[1,1]=0.715160
MatPass[1,2]= 0.072169
MatPass[2,0]= 0.019334
MatPass[2,1]= 0.119193
MatPass[2,2]=0.950227

# Blanc de reference (F7)
stdIllum = np.zeros(3)  
stdIllum[0]= 0.95041  
stdIllum[1]= 1.00000
stdIllum[2]= 1.08747

#################################
## Definition de l'image de test#
#################################
#V =255.0
##BGR
##Test Bleu Jaune
#mat2 = [[0.0,V,V]]
#mat1= [[V,0.0,0.0]]
#
##Test Rouge Vert
#mat3 = [[0.0,V,0.0]]
#mat4= [[0.0,0.0,V]]
#
##imgtest = np.zeros((100,300,3))
##imgtest[0:100,0:30]= mat1
##imgtest[0:100,30:60]= mat2
##imgtest[0:100,60:90]= mat1
##imgtest[0:100,90:120]= mat2
##imgtest[0:100,120:150]= mat1
##imgtest[0:100,150:180]= mat2
##imgtest[0:100,180:210]= mat1
##imgtest[0:100,210:240]= mat2
##imgtest[0:100,240:270]= mat1
##imgtest[0:100,270:300]= mat2
#
#
#
#imgtest = np.zeros((100,300,3))
#imgtest[0:50,0:30]= mat1
#imgtest[0:50,30:60]= mat2
#imgtest[0:50,60:90]= mat1
#imgtest[0:50,90:120]= mat2
#imgtest[0:50,120:150]= mat1
#imgtest[0:50,150:180]= mat2
#imgtest[0:50,180:210]= mat1
#imgtest[0:50,210:240]= mat2
#imgtest[0:50,240:270]= mat1
#imgtest[0:50,270:300]= mat2
#
#imgtest[50:100,0:30]= mat1
#imgtest[50:100,30:60]= mat2
#imgtest[50:100,60:90]= mat1
#imgtest[50:100,90:120]= mat2
#imgtest[50:100,120:150]= mat1
#imgtest[50:100,150:180]= mat2
#imgtest[50:100,180:210]= mat1
#imgtest[50:100,210:240]= mat2
#imgtest[50:100,240:270]= mat1
#imgtest[50:100,270:300]= mat2
#
###################################################################
####Ouverture de l'image et conrversion en 32bit#######
###################################################################()
#
##imag = cv2.imread("Food.0006.ppm",1)
#imag = cv2.imread("Flowers.0002.ppm",1)
##imag = cv2.imread("CapFood.png",1)
##imag = cv2.imread("test2.jpg",1)
#cv2.imshow('imgtest',imgtest)
#cv2.waitKey(0)
#imgtest=imgtest.astype(np.float32)
#imag=imag.astype(np.float32)
#
##Lab=RGBtoLAB(imgtest)
#Lab=RGBtoLAB(imag,MatPass, stdIllum)
#
###################################################################
########Passage de l'image dans l'espace Lab#######################
###################################################################
#
#
#
#
#
#
#
#
#cv2.imshow('L',Lab[:,:,0])
#cv2.waitKey(0)
#cv2.imshow('a',Lab[:,:,1])
#cv2.waitKey(0)
#cv2.imshow('b',Lab[:,:,2])
#cv2.waitKey(0)
#
###################################################################
###### Fonction de calcul de la difference#########################
###################################################################
#LabDiff=0
#LabDiff = diff(Lab,1,0)
#
#
#
#SphereCoord=CarthesianToSpheric(LabDiff)
#
#
#
#Test = SphericQuantif(SphereCoord,20,10)
#plt.figure()
#plt.plot(Test)
#
#L=SphereCoord[:,:,0]
#a=SphereCoord[:,:,1]
#b=SphereCoord[:,:,2]
a = Queue()
timer = Timer()  
timer.start()  
for i in np.arange(0,4):
    print i , ((2*np.pi)/4)*i
    Test = C2O("Food.0006.ppm", 1, ((2*np.pi)/4)*i, 4, 10, 5,a)
    plt.figure()
    plt.plot(Test)
timer.stop()  
print 'Temps sans parralelisation:', timer.interval  
Test = np.zeros((200,4))
Test1 = np.zeros(200)
#timer.start() 
#if __name__ == '__main__':
##    freeze_support()
#    q1 = Queue()
#    q2 = Queue()
#    q3 = Queue()
#    q4 = Queue()
#    p1 = Process(target=C2O,args=("Food.0006.ppm", 1, ((2*np.pi)/4)*0, 4, 10, 5,q1))
#    p2 = Process(target=C2O,args=("Food.0006.ppm", 1, ((2*np.pi)/4)*1, 4, 10, 5,q2))
#    p3 = Process(target=C2O,args=("Food.0006.ppm", 1, ((2*np.pi)/4)*2, 4, 10, 5,q3))
#    p4 = Process(target=C2O,args=("Food.0006.ppm", 1, ((2*np.pi)/4)*3, 4, 10, 5,q4))
#    
#    p1.start()
#    p2.start()
#    p3.start()
#    p4.start()
#    Test[:,0] = q1.get()
#    Test[:,1] = q2.get()
#    Test[:,2] = q3.get()
#    Test[:,3] = q4.get()
#    plt.figure()
#    for i in np.arange(0,200):
#        Test1[i] = np.mean(Test[i,:])
#    plt.plot(Test1)
#
#timer.stop()  
#print 'Temps avec parralelisation:', timer.interval 

#######################################################################################################################
#####################################################Affichage#########################################################
#######################################################################################################################
#fig = plt.figure()
#
#ax = fig.add_subplot(111, projection='3d')
##
##
#del SphericQuantif
#del SphereCoord
#del Lab
##
##
#ax.scatter3D(LabDiff[:,:,1], LabDiff[:,:,2], LabDiff[:,:,0],zdir='z')
#ax.set_xlim([-60, 60])
#ax.set_ylim([-60, 60])
#ax.set_zlim([-80, 80])
##ax.contourf(LabDiff[:,:,1], LabDiff[:,:,2], LabDiff[:,:,0], zdir='z',offset=np.min(LabDiff[:,:,0]), cmap='coolwarm')
##ax.contourf(LabDiff[:,:,1], LabDiff[:,:,2], LabDiff[:,:,0], zdir='y',offset=np.min(LabDiff[:,:,2]), cmap='coolwarm')
##ax.contourf(LabDiff[:,:,1], LabDiff[:,:,2], LabDiff[:,:,0], zdir='x',offset=-np.min(LabDiff[:,:,1]), cmap='coolwarm')
#cv2.destroyAllWindows()
#ax.set_xlabel('a')
#ax.set_ylabel('b')
#ax.set_zlabel('L')

#plt.show()
