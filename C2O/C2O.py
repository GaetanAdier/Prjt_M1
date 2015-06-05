# -*- coding: utf-8 -*-
#"""
#Created on Mon May 18 17:08:26 2015
#
#@author: etienne
#"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from multiprocessing import Process, Queue
import time  
import Constant as c
from sphinx_doc import genere_doc
from sphinx_doc import configure_doc

from docutils.core import publish_parts 
from PIL import Image
from skimage.color import rgb2xyz


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
    
    Function for the calculation of a difference of color on the whole image, following the :math:`\Delta` vector
    caracterized here by dX and dY (calculate from the radix and the norm of the vector).
    The diffenrence between the image and it's shifted copy is computed as shown below 
    
    |
    
    .. image:: ColorDiff.png 
       :width: 400pt
       :align: center
       
    |
    
    This function is called as shown below :
            
    .. code-block:: python
       :emphasize-lines: 3,5
    
       imgDiff = diff(img,dX,dY)
      
      
      
        
    :param img: A N-dimensions matrix containing the image on which you need the C2O feature calculation.
    :type img: np.ndarray    
    :param dX: Number of pixel to shift on the X axis.
    :type dX: float   
    :param dY: Number of pixel to shift on the Y axis.
    :type dY: float 
    
    :return: The difference of color image 
    :rtype: ndarray
    
    """
    
        
    
    rows,cols , plans = img.shape
    
    #Image shifting
    M = np.float32([[1,0,-dX],[0,1,dY]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    NegY=0 
    NegX=0
    

    
    cv2.imshow('Image dorigine decalee',dst)
    cv2.waitKey(0)
#    
#    
    reMatimg=np.zeros([rows-dY,cols-dX,plans])
    reMatimg=img[dY:rows,0:cols-dX,:]
    
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
#    reMatdst=np.zeros([rows-dY,cols-dX,plans])
#    reMatdst=dst[dY-NegY:rows-NegY,NegX:cols-dX+NegX,:]
    reMatdst=np.zeros([rows-dY,cols-dX,plans])
    reMatdst=dst[dY:rows,0:cols-dX,:]
    
    print dY-NegY, rows-NegY, NegX ,cols-dX+NegX
    
    cv2.imshow('Image de destination decale resizee',reMatdst)
    cv2.waitKey(0)
    
#    reMatimg=np.zeros([rows-dY,cols-dX,plans])
#    reMatimg=img[dY+NegY:rows+NegY,NegX:cols+dX+NegX,:] 
    reMatimg=np.zeros([rows-dY,cols-dX,plans])
    reMatimg=img[dY:rows,0:cols-dX,:]
    
    cv2.imshow('image non decalee resizee',reMatimg)
    cv2.waitKey(0)
    
    Res = np.zeros([rows-dY,cols-dX,plans])

    # Difference computing
    Res= reMatimg-reMatdst
    

    
    cv2.imshow('b',Res)
    cv2.waitKey(0)    
    
    # Uncomment to display the coocurence matrix
    

 

    return Res
    
    
    



def RGBtoLAB(imag, MatPass, stdIllum, gamma):
    
    ur"""
    
    
    Function for computing the transformation from RGB to :math:`L^*a^*b^*` space. This function use the transition RGB to XYZ space to compute the :math:`L^*a^*b^*` space.
    
    This transformation is computed as show below : 
    
    .. math::
        A=\begin{pmatrix}X_r&X_g&X_b\\Y_r&Y_g&Y_b\\ Z_r&Z_g&Z_b\end{pmatrix}
       
       
    .. math::
        \begin{pmatrix}X\\Y\\Z\end{pmatrix}=A*\begin{pmatrix}R\\G\\B\end{pmatrix}
    
    With :math:`A` depending on the RGB space considered for the image (all possible matrix are available in :py:class:`Constant.MatPass`).
    
    After this transformation, the :math:`L^*a^*b^*` is computed following these formulas : 

    .. math::
      L^*=  \left \{\begin{array}{l}116*(\frac{Y}{Y_0})^\frac{1}{3}-16~~~~if \frac{Y}{Y_0}>0.008856\\903.3*(\frac{Y}{Y_0})~~~~~~~~~~if \frac{Y}{Y_0}<0.008856\\\end{array}\right .
    
    .. math::
      a^*=500*\begin{bmatrix}f(\frac{X}{X_0})-f(\frac{Y}{Y_0})\end{bmatrix}

    .. math::
      b^*=200*\begin{bmatrix}f(\frac{Y}{Y_0})-f(\frac{Z}{Z_0})\end{bmatrix}



    To make proper the RGB to XYZ transformation, it's make an inverse companding to counteract the non-linearity of the RGB. 
    
    The :math:`\gamma` parameter associate with each RGB space is specified in the :py:mod:`Constant` module documentation (Associated with the transformation matrix).
    
        
    This function is called as shown below :
    
    .. code-block:: python
       :emphasize-lines: 3,5
    
       Lab = RGBtoLAB(imag, MatPass, stdIllum, gamma)
    
   
    
    
    :param gamma: Inverse companding parameter :math:`\gamma`.
    :type gamma: float
    :param imag: A matrix containing the image on which you need the C2O feature calculation.
    :type imag: np.ndarray
    :param MatPass: The transition matrix for the BGR to XYZ transformation (set at '0' to have the default value)
    :type MatPass: np.ndarray
    :param stdIllum: The standard illuminant choosen for the BGR to XYZ transformation (set at '0' to have the default value).
    :type stdIllum: np.ndarray
    
    :return: The coocurence matrix in spherical coordinates.
    :rtype: np.ndarray    
    
    

        
    """
    
    
    rows,cols, plans = imag.shape

 
    imag=imag.astype(np.float32)
    
    # Set values in the range [0,1]
    imag = imag/255.0
    
    #Inverse companding on the RGB channel
    imag = imag**gamma
    
    
    XYZ=np.zeros((rows,cols,3))
    XBIN=np.zeros((rows,cols))
    YBIN=np.zeros((rows,cols))
    ZBIN=np.zeros((rows,cols))
    Lab=np.zeros((rows,cols,3))
    Lab1=np.zeros((rows,cols,3))

    


#     Transformation throught the XYZ space
    XYZ[:,:,0] = (imag[:,:,2]*MatPass[0,0] + imag[:,:,1]*MatPass[0,1] + imag[:,:,0]*MatPass[0,2])/stdIllum[0]
    XYZ[:,:,1] = (imag[:,:,2]*MatPass[1,0] + imag[:,:,1]*MatPass[1,1] + imag[:,:,0]*MatPass[1,2])/stdIllum[1]
    XYZ[:,:,2] = (imag[:,:,2]*MatPass[2,0] + imag[:,:,1]*MatPass[2,1] + imag[:,:,0]*MatPass[2,2])/stdIllum[2]
    
#    XYZ = rgb2xyz(imag)
    
    XYZ1 = XYZ
    
    # Thresholding matrix
    XBIN[XYZ[:,:,0]>0.008856]=1.0
    YBIN[XYZ[:,:,1]>0.008856]=1.0
    ZBIN[XYZ[:,:,2]>0.008856]=1.0
     
    
    

            

#   #L               Cas ou Y>0.008856                                      Cas ou Y<0.008856
    Lab1[:,:,0] = ((116*((XYZ[:,:,1])*YBIN)**(1.0/3.0))-16)             +(903.3*((XYZ[:,:,1])* np.abs(YBIN-1)))
#   #A                      X>0.008656                      Y>0.008856                                      X<0.008856                                          Y<0.008856
    Lab1[:,:,1] = 500*(((XBIN*(XYZ[:,:,0])**(1.0/3.0))-(YBIN*(XYZ[:,:,1])**(1.0/3.0)))            +(((np.abs(XBIN-1)*(XYZ[:,:,0])*(7.787))+(16.0/116.0))-((np.abs(YBIN-1)*(XYZ[:,:,1])*(7.787))+(16.0/116.0))))
#   #B                      Y>0.008656                      Z>0.008856                                      Y<0.008856                                          Z<0.008856
    Lab1[:,:,2] = 200*(((YBIN*(XYZ[:,:,1])**(1.0/3.0))-(ZBIN*(XYZ[:,:,2])**(1.0/3.0)))            +(((np.abs(YBIN-1)*(XYZ[:,:,1])*(7.787))+(16.0/116.0))-((np.abs(ZBIN-1)*(XYZ[:,:,2])*(7.787))+(16.0/116.0))))


    return Lab1
    
    

def CarthesianToSpheric(xyz):
    ur"""
    
    
    
    Function for computing spherical coordinates from carthesian ones.
    
    
    This function computes this transformation following the formulas shown below : 
    

    .. math::
      r = \sqrt{x^2+y^2+z^2}
      
      
    .. math::
      \phi=\arctan(\frac{y}{x})
    
    .. math::
      \theta = \arccos(\frac{z}{r}) 
      
    These parameters are corresponding to :
    
     * r : The norm of the vector formed by the point to convert in spherical coordinate and the origin point.
     * :math:`\alpha`: The radix between the :math:`(L^*b^*)` plan and the vector.
     * :math:`\beta`: The radix between the :math:`(L^*a^*)` plan and the vector.
     
    |
    
    .. image:: Spherical_Coordinates.png 
       :align: center
       
    |
    
    This function is called as shown below :
    
    .. code-block:: python
       :emphasize-lines: 3,5
    
       SphereCoords=CarthesianToSpheric(xyz)
     
    :param xyz: The 3-D matrix in carthesian coordinates.
    :type xyz: np.ndarray
    
    :return: The 3-D matrix in spherical coordinates.
    :rtype: np.ndarray
    

      
    .. note:: The fonction return the radix values :math:`\alpha` in range :math:`[0,2 \pi]` and :math:`\beta` in range :math:`[-\frac{\pi}{2}, \frac{\pi}{2}]`
      
    """

    
    
    rows,cols, plans = xyz.shape
    Spheric=np.zeros((rows,cols,plans))
    
    Spheric[:,:,0] = np.sqrt((xyz[:,:,0]**2)+(xyz[:,:,1]**2)+(xyz[:,:,2]**2))
    Spheric[:,:,1] = np.arctan2(xyz[:,:,1],xyz[:,:,0])
    Spheric[:,:,2] = np.arctan2(np.sqrt((xyz[:,:,0]**2)+(xyz[:,:,1]**2)),xyz[:,:,2])
    
    # Remise a l'echelle des valeurs d'angles 
    Spheric[:,:,1][Spheric[:,:,1]<0]=Spheric[:,:,1][Spheric[:,:,1]<0]+(2*np.pi)
    Spheric[:,:,2]=  Spheric[:,:,2] - np.pi/2
    return Spheric




def SphericToCartesian(Spheric):
    """
    
    
    Function for computing carthesian coordinates from spherical ones.
    
    
    
    
    :param Spheric: The coocurence matrix in spherical coordinates.
    :type Spheric: np.ndarray
    
    :return: The coocurence matrix in carthesian coordinates.
    :rtype: np.ndarray
    
    
    """
    

    
    
    rows,cols, plans = Spheric.shape
    Cartesian=np.zeros((rows,cols,plans))
    
    Cartesian[:,:,0] = Spheric[:,:,0] * np.sin(Spheric[:,:,2]) *  np.cos(Spheric[:,:,1])
    Cartesian[:,:,1] = Spheric[:,:,0] * np.sin(Spheric[:,:,2]) *  np.sin(Spheric[:,:,1])
    Cartesian[:,:,2] = Spheric[:,:,0] * np.cos(Spheric[:,:,2])
    
    
    return Cartesian
 


def SphericQuantif(C2OMat, NE, Nalpha, Nbeta):
    ur"""
    
    
    Function for computing the spherical quantization for the C2O signature.
    
    The C2O coocurence matrix is really large in term of the number of values and it is not really easy to compare with each other
    because of the 3-dimensions. So to solve it, the matrix is quantisize by computing this sphérical quantization to obtain a vector in 1-D.
    
    
    .. image:: QuantificationSpherique.png 
       :align: center
    
    This function is called as shown below :
    
    .. code-block:: python
       :emphasize-lines: 3,5
    
       SigC2O = SphericQuantif(C2OMat, NE, Nalpha, Nbeta)
    
    :param C2OMat: The coocurence matrix in spherical coordinates.
    :type C2OMat: np.ndarray
    :param Nalpha: Number of intervals considered for the signature calculation on the :math:`\alpha` component.
    :type Nalpha: float
    :param Nbeta: Number of intervals considered for the signature calculation on the :math:`\beta` component.
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

    print 9.0/(NE-1)
    
    for i in np.arange(0,NE):
        BinE=np.zeros((rows,cols))
        BinE[(C2OMat[:,:,0]>=(9.0/(NE-1))*i)&(C2OMat[:,:,0]<=(9.0/(NE-1))*(i+1))]=1
        if i > NE-2 :
            BinE[(C2OMat[:,:,0]>=(9.0/(NE-1))*i)]=1
   
        for j in np.arange(0,Nbeta):
            BinBeta=np.zeros((rows,cols)) #Beta va de - pi/2 a pi/2 et pas de 8 a pi
            BinBeta[(BinE==1)&(C2OMat[:,:,2]>=((-np.pi/2)+((np.pi/Nbeta)*j)))&(C2OMat[:,:,2]<=((-np.pi/2)+((np.pi/Nbeta)*(j+1))))]=1
            
          
            for k in np.arange(0,Nalpha):
                BinAlpha=np.zeros((rows,cols))
                BinAlpha[(BinBeta==1)&(C2OMat[:,:,1]>=(((2.0*np.pi/Nalpha)*k)-(np.pi/(Nalpha))))&(C2OMat[:,:,1]<=(((2.0*np.pi/Nalpha)*(k+1))-(np.pi/(Nalpha))))]=1

                SigC2O[n]=np.sum(BinAlpha)
                
                n = n+1
                
    return SigC2O
    
    
    
def C2O(image, NormDelta, RadDelta, NE, Nalpha, Nbeta, SigC2Ot):
    ur"""
    
    
    Function for the :math:`C_2O` feature calculation.
    
    The aim of this function is to compute a description feature of a color image which includes the color and texture information. 
    The attempted result is to obtain on unique vector for the whole image which characterize the best its content.  
    
    This function uses the following formulas to calculate the number of pixel to shift on the image for the difference of color (dX and dY)
    
    .. math::
      \sin\theta=dY/\|\Delta\| 
    .. math::
      dY = \sin\theta * \|\Delta\| 
    .. math::
      \cos\theta=dX/\|\Delta\| 
    .. math::
      dX = \cos\theta * \|\Delta\|
      
      
    After the difference computing, the result obtained is the :math:`C_2O` coocurence matrix : 
     
    .. image:: C2OMatrix.png 
       :width: 400pt
       :align: center
     
    (Matrix computed from a 128*128 pixel sample of the Food0006.ppm image from VISTEX database)
 
    From this matrix, the function extract the :math:`C_2O` signature descriptor by a spherical quantization (see:SphereQuantif()). The result obtained from the previous exemple is :
    
    .. image:: C2OSig.png 
       :width: 400pt
       :align: center    
    
    This function is called as shown below :
    
    .. code-block:: python
       :emphasize-lines: 3,5
    
       SigC2O = C2O(image, NormDelta, RadDelta, NE, Nalpha, Nbeta, SigC2Ot)
    
    :param image: Path directory of the image on which you need the :math:`C_2O` feature calculation.
    :type image: string
    :param NormDelta: Norm of :math:`\Delta` vector for the image color diference.
    :type NormDelta: float
    :param RadDelta: Radix of :math:`\Delta` vector for the image color diference.
    :type RadDelta: float
    :param Nalpha: Number of intervals considered for the signature calculation on the :math:`\alpha` component.
    :type Nalpha: float
    :param Nbeta: Number of intervals considered for the signature calculation on the :math:`\beta` component.
    :type Nbeta: float
    :param SigC2Ot: Return parameter for the parralelise version.
    :type SigC2Ot: queue
    
    :return: The signature of the image in the RadLambda orientation and at the NormLambda distance for color difference in a vector of NE*Nalpha*Nbeta
    :rtype:  np.ndarray
    
    This function use :py:func:`SphericQuantif` 
    
    
    """
    

#    
    
    dX = 0
    dY = 0    
    

    
    imag = Image.open(image)
    
    imag=np.array(imag)
    
    
    
    
    Lab = np.zeros(np.shape(imag))
#    imag=imag.astype(np.float32)
    #Transformation throught the Lab space    
    Lab=RGBtoLAB(imag,c.MatPass.AdobRGB, c.stdIlluminant.D65, 2.2)
    
    
    LabDiff=0
    # Calculation of the shifting parameter
    dX = np.round(np.cos(RadDelta)*NormDelta)
    dY = np.round(np.sin(RadDelta)*NormDelta)
    
    
    
    print dX , dY
    
#    Calculation of the difference of color
    LabDiff = diff(Lab,dX,dY)
    rows,cols, plans = LabDiff.shape
#    print np.max(Lab[:,:,0]) , np.min(Lab[:,:,0])
#    print np.max(Lab[:,:,1]) , np.min(Lab[:,:,1])
#    print np.max(Lab[:,:,2]) , np.min(Lab[:,:,2])
    # Transformation in spherical coordinates
    SphereCoord=CarthesianToSpheric(LabDiff)
    

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter3D(LabDiff[:,0:cols,2], LabDiff[:,0:cols,1], LabDiff[:,0:cols,0],zdir='z')
    ax.set_xlim([-80, 80])
    ax.set_ylim([-80, 80])
    ax.set_zlim([-80, 80])
    
    ax.contourf(LabDiff[:,:,2], LabDiff[:,:,1], LabDiff[:,:,0], zdir='z',offset=-80, cmap='coolwarm')
    ax.contourf(LabDiff[:,:,2], LabDiff[:,:,1], LabDiff[:,:,0], zdir='y',offset= 80, cmap='coolwarm')
    ax.contourf(LabDiff[:,:,2], LabDiff[:,:,1], LabDiff[:,:,0], zdir='x',offset=-80, cmap='coolwarm')
#    cv2.destroyAllWindows()
    ax.set_xlabel(ur"$\Delta$"+"b")
    ax.set_ylabel(ur"$\Delta$"+"a")
    ax.set_zlabel(ur"$\Delta$"+"L")

    ax.set_title(ur"$C_2O$"+" matrix")
    # Calculation of the signature by the spherical quantization
    SigC2O = SphericQuantif(SphereCoord,NE ,Nalpha,Nbeta)
    
    # Assignation of the result on the queue for the parralelized version
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
V =255.0
##BGR
##Test Bleu Jaune
mat2 = [[V,V,0.0]]
mat1= [[0.0,0.0,V]]

#Test Rouge Vert
#mat2 = [[V,0.0,0.0]]
#mat1= [[0.0,V,0.0]]
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
imgtest = np.zeros((100,300,3))
imgtest[0:50,0:30]= mat1
imgtest[0:50,30:60]= mat2
imgtest[0:50,60:90]= mat1
imgtest[0:50,90:120]= mat2
imgtest[0:50,120:150]= mat1
imgtest[0:50,150:180]= mat2
imgtest[0:50,180:210]= mat1
imgtest[0:50,210:240]= mat2
imgtest[0:50,240:270]= mat1
imgtest[0:50,270:300]= mat2

imgtest[50:100,0:30]= mat1
imgtest[50:100,30:60]= mat2
imgtest[50:100,60:90]= mat1
imgtest[50:100,90:120]= mat2
imgtest[50:100,120:150]= mat1
imgtest[50:100,150:180]= mat2
imgtest[50:100,180:210]= mat1
imgtest[50:100,210:240]= mat2
imgtest[50:100,240:270]= mat1
imgtest[50:100,270:300]= mat2
#

#cv2.imwrite('ImgTestBJRGB.png',imgtest)

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





##################################################################
##########################doc#####################################
#configure_doc()
genere_doc()

#publish_parts(diff.__doc__) 


a = Queue()
#timer = Timer()  
#timer.start()  
#for i in np.arange(0,4):
#    print i , ((2*np.pi)/4)*i
#    Test = C2O("grayimage.jpg", 1, ((2*np.pi)/4)*i, 4, 10, 5,a)
#    plt.figure()
#    plt.plot(Test)
#timer.stop()  
#print 'Temps sans parralelisation:', timer.interval  
#Test = np.zeros((200,4))
#Test1 = np.zeros(200)
###################################################################
##################################################################
##################################################################

#Test = C2O("000066.bmp", 1, 0, 4, 8, 8,a)
#plt.figure()
#plt.plot(Test)
#plt.title(ur"$C_2O$"+" signature")



########################################################################
#########Validation quantif spherique###################################
########################################################################
#NumIntervE = 0.0
#NumIntervAlpha = 4.0
#NumIntervBeta = 2.0
#
#
#E =  np.abs(np.random.randn(30,30))*(3.0/4.0)
#E = E + (3*NumIntervE)
#alpha = np.random.randn(30,30)*(np.pi/(8*4))
#alpha = alpha + ((NumIntervAlpha*2.0*np.pi/8.0))
#beta = np.abs(np.random.randn(30,30))*(np.pi/(8*4))
#beta = beta - ((4.0-NumIntervBeta)*np.pi/8.0)
#
#SphereCoord = np.zeros((30,30,3))
#SphereCoord[:,:,0]= E
#SphereCoord[:,:,1]= alpha
#SphereCoord[:,:,2]= beta
#Test = SphericQuantif(SphereCoord,4,8,8)
##
###
###
##plt.figure()
##plt.plot(Test)
##plt.title("C2O Signature")
#
#Mat = SphericToCartesian(SphereCoord)

#plt.title('Signature de image test bleu/Jaune')

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
##    for i in np.arange(0,200):
##        Test1[i] = np.mean(Test[i,:])
##    plt.plot(Test1)
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
##del SphericQuantif
##del SphereCoord
##del Lab
##
##
#ax.scatter3D(Mat[:,:,1], Mat[:,:,2], Mat[:,:,0],zdir='z')
##ax.set_xlim([-60, 60])
##ax.set_ylim([-60, 60])
##ax.set_zlim([-80, 80])
#ax.set_xlim([-2, 2])
#ax.set_ylim([-2, 2])
#ax.set_zlim([-2, 2])
#ax.set_title('C2O matrix')
#ax.set_xlabel(ur"$\Delta$"+"b")
#ax.set_ylabel(ur"$\Delta$"+"a")
#ax.set_zlabel(ur"$\Delta$"+"L")
#ax.contourf(Mat[:,:,1], Mat[:,:,2], Mat[:,:,0], zdir='z',offset=-2, cmap='coolwarm')
#ax.contourf(Mat[:,:,1], Mat[:,:,2], Mat[:,:,0], zdir='y',offset= 2, cmap='coolwarm')
#ax.contourf(Mat[:,:,1], Mat[:,:,2], Mat[:,:,0], zdir='x',offset=-2, cmap='coolwarm')
##ax.plot([-60,60], [0,0], [0,0])
##ax.plot([0,0], [-60,60], [0,0])
##ax.plot([0,0], [0,0], [-80,80])
#ax.plot([-1,1], [0,0], [0,0])
#ax.plot([0,0], [-1,1], [0,0])
#ax.plot([0,0], [0,0], [-1,1])
cv2.destroyAllWindows()


#plt.show()
