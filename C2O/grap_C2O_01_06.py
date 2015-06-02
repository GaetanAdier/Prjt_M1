#!/usr/bin/env python

__doc__ = """
Program for dif image graphics
Date: Mai 2015

    - graphe: graphics of difference in LAB esphere coordinates, 
   
"""
from PIL import Image
import numpy as np
from skimage.color import rgb2lab

from C2O import SphericQuantif

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


###########################################################################################
def graph (imat):      
        #colour=['b','r','g','y','m','c']        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
          
        ax.scatter3D(imat[:,:,2],imat[:,:,1],imat[:,:,0])# plot the 3D graph
       
        ax.set_xlim([-60, 60])
        ax.set_ylim([-60, 60])
        ax.set_zlim([-80, 80])
         #plot de proyection in each axe
        ax.contourf(imat[:,:,2],imat[:,:,1],imat[:,:,0], zdir='x',offset=-60,cmap=cm.coolwarm)
        ax.contourf(imat[:,:,2],imat[:,:,1],imat[:,:,0], zdir='y',offset=60,cmap=cm.coolwarm)
        ax.contourf(imat[:,:,2],imat[:,:,1],imat[:,:,0], zdir='z',offset=-80,cmap=cm.coolwarm)                                                               
                
        plt.show()
###########################################################################
# end function graphespherique


############################################################################	
##===============================================================================
##        The call main
##===============================================================================

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

if __name__ == "__main__":
        
        #waynom="C:/data/ima/Outex_TC_00014/images/000020" # way of image file   
        waynom="000066"
        wayext=".bmp" #write the extension of file
        way1=waynom+wayext # the way and their extention
        imat=Image.open(way1)  #image way
        [fil,col]=imat.size
        print fil
        print col
        ima1=imat.crop((0,0,fil-1,col)) #refence image is croped 
        ima2=imat.crop((1,0,fil,col)) #we do a copy of image a pixel desfased
        arrayima1=np.array(ima1) #image en array 
        imalab1 = rgb2lab(arrayima1) #convert an image RGB to CIE Lab
#        imalab1 = RGBtoLAB(arrayima1,MatPass,stdIllum)
        arrayima2=np.array(ima2)# image en array 
        imalab2 = rgb2lab(arrayima2) #convert an image RGB to CIE Lab
#        imalab2 = RGBtoLAB(arrayima2,MatPass,stdIllum)
        imate=imalab1-imalab2
      
        graph(imate)
        
        test = SphericQuantif(imate, 4, 20, 10)
        
        

        
        plt.figure()
        
        plt.plot(test)
        plt.title('Test')
        


                




