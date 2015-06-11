# -*- coding: utf-8 -*-
"""
This module gathering all the functions necessary to process the different descriptor computing.
"""

#import cv2
import numpy as np
import time  
import Constant as c

from mpl_toolkits.mplot3d import Axes3D 
from multiprocessing import Process, Queue
from sphinx_doc import genere_doc
from sphinx_doc import configure_doc
from docutils.core import publish_parts 
from PIL import Image
from collections import Counter


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
    

#    

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

    reMatdst=np.zeros([rows-dY,cols-dX,plans])
    reMatdst=dst[dY:rows,0:cols-dX,:]
    

    reMatimg=np.zeros([rows-dY,cols-dX,plans])
    reMatimg=img[dY:rows,0:cols-dX,:]

    
    Res = np.zeros([rows-dY,cols-dX,plans])

    # Difference computing
    Res= reMatimg-reMatdst
    

 
    
    # Uncomment to display the coocurence matrix
    
    del dst
    del img

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
    Spheric[:,:,1][Spheric[:,:,1]<0]=Spheric[:,:,1][Spheric[:,:,1]<0]+(2.0*np.pi)
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
    
    
    .. image:: QuantificationSphericToHistCorrige.png 
       :width: 500pt
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

    for i in np.arange(0,Nbeta):
        BinBeta=np.zeros((rows,cols))
        BinBeta[(C2OMat[:,:,2]>=((-np.pi/2)+((np.pi/Nbeta)*i)))&(C2OMat[:,:,2]<=((-np.pi/2)+((np.pi/Nbeta)*(i+1))))]=1

        for j in np.arange(0,NE):

            BinE=np.zeros((rows,cols))
            BinE[(BinBeta==1)&((C2OMat[:,:,0]>=(9.0/(NE-1))*j)&(C2OMat[:,:,0]<=(9.0/(NE-1))*(j+1)))]=1
  

            BinE[(j > NE-2)&(BinBeta==1)&(C2OMat[:,:,0]>=(9.0/(NE-1))*j)]=1
            BinAlpha=np.zeros((rows,cols))
            BinAlpha[(BinE==1)&(((C2OMat[:,:,1]>=((2.0*np.pi)-(np.pi/(Nalpha))))&(C2OMat[:,:,1]<=(2.0*np.pi)))|((C2OMat[:,:,1]<=(np.pi/(Nalpha)))&((C2OMat[:,:,1]>=0))))]=1
            SigC2O[n]=np.sum(BinAlpha)
            n = n+1

            for k in np.arange(1,Nalpha):
                BinAlpha=np.zeros((rows,cols))

                
                BinAlpha[(BinE==1)&(C2OMat[:,:,1]>=(((2.0*np.pi/Nalpha)*k)-(np.pi/(Nalpha))))&(C2OMat[:,:,1]<=(((2.0*np.pi/Nalpha)*(k+1))-(np.pi/(Nalpha))))]=1                
                
                
#      
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
    
    timer = Timer()  

#    
    
    dX = 0
    dY = 0    
    

    

    
    
    
    Lab = np.zeros(np.shape(image))

    #Transformation throught the Lab space    
 
    Lab=RGBtoLAB(image,c.MatPass.AdobRGB, c.stdIlluminant.D65, 2.2)

    
    LabDiff=0
    # Calculation of the shifting parameter
    dX = np.round(np.cos(RadDelta)*NormDelta)
    dY = np.round(np.sin(RadDelta)*NormDelta)
    
    
    
    
#    Calculation of the difference of color
    LabDiff = diff(Lab,dX,dY)


    rows,cols, plans = LabDiff.shape
    # Transformation in spherical coordinates

    SphereCoord=CarthesianToSpheric(LabDiff)


    SigC2O = SphericQuantif(SphereCoord,NE ,Nalpha,Nbeta)

    
    # Assignation of the result on the queue for the parralelized version
    SigC2Ot.put(SigC2O)
    
    return SigC2O 


def C2OPatch(image,dE,dAlpha,dBeta):
    ur"""
    
    
    Function for the :math:`C_2O` feature calculation relative to keypoint detection (from openCV's SIFT).

    This function uses the :py:func:`C2O` function on patch of 64*64 pixel around each keypoint.
      
    
    This function is called as shown below :
    
    .. code-block:: python
       :emphasize-lines: 3,5
    
       MatSigC2O = C2OPatch(image,kp)
    
    :param image: Path directory of the image on which you need the :math:`C_2O` feature calculation.
    :type image: string
    :param kp: Keypoints matrix from OpenCv's SIFT keypoint detection.
    :type kp: OpenCV keypoint structure array

    
    :return: The C2O signatures for each keypoint in an array.
    :rtype:  np.ndarray
    
    This function use :py:func:`C2O` 
    
    
    """   
        
    
    imag = Image.open(image)
    imag=np.array(imag)
    a = Queue()
    
    rows,cols, plans = imag.shape
    

    gray= cv2.cvtColor(imag,cv2.COLOR_RGB2GRAY)
    
    sift = cv2.SIFT()
    kp = sift.detect(gray,None)
    
    size = dE*dAlpha*dBeta
    
    matC2OPatch = np.zeros((len(kp),size))
    mat_kp=np.zeros([len(kp),2])  
    for i in range(len(kp)):
        mat_kp[i][0]=np.round(kp[i].pt[0])
        
    for j in range(len(kp)):
        mat_kp[j][1]=np.round(kp[j].pt[1])
    matkpret = np.zeros(np.shape(mat_kp))
    n = 0
    
    

    for i in np.arange(0,len(kp)-1):

        if (((mat_kp[i][0]-31)>0) & ((mat_kp[i][0]+32)<rows-1) & ((mat_kp[i][1]-31)>0) & ((mat_kp[i][1]+32)<cols-1)):

            matkpret[n][0] = mat_kp[i][0]
            matkpret[n][1] = mat_kp[i][1]
            matC2OPatch[n,:]=C2O(imag[mat_kp[i][0]-31:mat_kp[i][0]+32,mat_kp[i][1]-31:mat_kp[i][1]+32], 1, 0, 4, 6, 3,a)
            n = n+1
            
  
    matC2OPatch = np.resize(matC2OPatch,(n,size))
    matkpret = np.resize(matkpret,(n,2))
    return matkpret , matC2OPatch
    
def SIFT(img):
    
    """
    
    Function for computing the K-means method.
    
    This function find center of vectors and groups input samples
    
    
    This function is called as shown below :
    
    .. code-block:: python
       :emphasize-lines: 3,5
    
       kp,desc = SIFT(list_path_img[i-1])
    
    :param list_path_img: List which contains all the path of the images present in the database.
    :type Vectors: list
    
    :return kp: Matrix which contains the Key-points in one image
    :rtype: nd.array
    :return desc: Matrix which contains the descriptors for one image
    :rtype: nd.array
    
    
    """
    
    img_trait=cv2.imread(img)
    
    grayimage=cv2.cvtColor(img_trait, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("grayimage.jpg",grayimage)
    
    sift = cv2.SIFT(0,3,0.04,10,1.6)
    kp,des = sift.detectAndCompute(grayimage,None)
    
    return kp,des