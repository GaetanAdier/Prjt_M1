# -*- coding: utf-8 -*-
"""
Created on Mon May 04 15:51:37 2015

@author: RTMA
"""

import os
import function_sift as fc
from multiprocessing import Process, Queue
import time

path_images = "C:\\Users\RTMA\\Desktop\\Nouveau dossier\\khalil"#PlantCLEF2015TrainingData\\train"   
path_work = "C:\\Users\\RTMA\\Desktop\\Nouveau dossier"
descriptor = "SIFT_2"


if not(os.path.isdir(path_work)):
    os.mkdir(path_work)
    
if not(os.path.isdir(path_images)):
    print("path for work on images doesn't exist")
    exit(0)
    
#sans parallelisme
start = time.time()    
#fc.descript(path_work, descriptor, path_images)
#print(time.time()-start)

#parallelisme
if __name__ == '__main__':
    q = Queue()
    p = Process(target=fc.descript, args=(path_work,descriptor,path_images))
    p.start()
    print(time.time()-start)
    
