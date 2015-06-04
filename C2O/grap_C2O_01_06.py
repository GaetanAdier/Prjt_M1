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
        ax.set_title('Matrice C2O fonction Armando')
        ax.set_ylim([-60, 60])
        ax.set_zlim([-80, 80])
        ax.set_xlim([-60, 60])
#         #plot de proyection in each axe
#        ax.contourf(imat[:,:,2],imat[:,:,1],imat[:,:,0], zdir='x',offset=-60,cmap=cm.coolwarm)
#        ax.contourf(imat[:,:,2],imat[:,:,1],imat[:,:,0], zdir='y',offset=60,cmap=cm.coolwarm)
#        ax.contourf(imat[:,:,2],imat[:,:,1],imat[:,:,0], zdir='z',offset=-80,cmap=cm.coolwarm)                                                               
                
        plt.show()
###########################################################################
# end function graphespherique


############################################################################	
##===============================================================================
##        The call main
##===============================================================================



if __name__ == "__main__":
        
        #waynom="C:/data/ima/Outex_TC_00014/images/000020" # way of image file   
        waynom="5"
        wayext=".jpg" #write the extension of file
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
        

        


                




