# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 17:33:54 2015

@author: gaetan
"""

import os
import glob
import numpy as np
import ntpath

import Descripteur as Desc
import Classif as Class
import getClassID as gtID


def Process_flow(path_work, name_desc, path_images, nb_word, sizeDesc, nb_images = "ALL", start_img = 1):
    
    """
    Function called in the main, she do all the chain process. Here you can choose your path directory in function of the differents descriptors you want to compile. You will found the next parameters \:

    
    :param path_work: Path where the user wants to create folders for the descriptors.
    :type path_work: string
    :param name_desc: Name of descriptor choosen
    :type name_desc: string
    :param path_images: Path of workspace: pictures should be in this folder.
    :type path_images: string
    :param nb_word: Number of words you want in the Bag of Words.
    :type nb_word: int
    :param sizeDesc: size of the descriptor. For example: the SIFT descriptor has a size = 128
    :type sizeDesc: int
    :param nb_images: Number of pictures to process. Default : ALL
    :type nb_images: int
    :param start_img: Number of the start picture (if you don't want to process all the images) . Default : 1
    :type start_img: int
    
    :return res: result of the quantization, one line = the quantization for this images
    :rtype: nd.array
    :return ID[70:100]: ID of the pictures take in the xml files (if the quantization is perfect res = ID[70:100])
    :rtype: nd.array
    :return: List of the path of the images to process (in the order of process)
    :rtype: list   
     
    To compute the SIFT descriptors you have to put in commentary the next line (103)
    
    .. code-block:: python
       :emphasize-lines: 3,5
    
       mat_kp, desc = CO.C2OPatch(list_path_img[i-1], 4, 6, 3)
    
    To compute the C2O descriptors you have to put in commentary the next line (104)
    
    .. code-block:: python
       :emphasize-lines: 3,5
    
       kp,desc = SIFT(list_path_img[i-1])
       
    For launch the compilation of this function, put this line : 
    
    .. code-block:: python
       :emphasize-lines: 3,5
    
       ID_classif, ID_know, img = fc.Process_flow(path_work, descriptor, path_images, nb_word, sizeDesc)
        
    >>> 
    >>> 
     
    """
    
    if not(os.path.isdir(path_work)):
        os.mkdir(path_work)
    
    if not(os.path.isdir(path_images)):
        print("path for work on images doesn't exist")
        exit(0)
    
    #creation du dossier contenant les donnnées de chaque images pour le descripteur choisi
    path_desc = "%s\\%s" % (path_work, name_desc) 
    
    if not(os.path.isdir(path_desc)):
        os.mkdir(path_desc)

    #creation d'une variable contenant les chemins de toutes les images contenues dans le dossier des images à traiter
    temp =  "%s\\*.jpg" % (path_images)
    temp2 =  "%s\\*.xml" % (path_images)
    list_path_img = glob.glob(temp)   
    list_path_img2 = glob.glob(temp2)   
    nb_img = len(list_path_img)
    
    #permet de parcourir toutes ou un nombre d'image définis par l'utilisateur de manière automatique    
    if(nb_images == "ALL"):
        end_img = nb_img - start_img + 1
    else : 
        end_img = nb_images + start_img
    
    all_desc = np.zeros((0,sizeDesc))
    nb_kp_per_img = np.zeros((0,1))
    sig = np.zeros((0,nb_word)) 
    
    ID = np.zeros(0)
    #application du descripteur choisit sur les images
    for i in range(start_img, (end_img + 1)): 
        kp,desc = Desc.SIFT(list_path_img[i-1])
#        mat_kp, desc = Desc.C2OPatch(list_path_img[i-1], 4, 6, 3)
        
        temp =  gtID.GetClassID(list_path_img2[i-1])
        ID = np.append(ID, temp)

        
        
    #Enregistrement des descripteurs dans fichiers txt
        filename=ntpath.basename(list_path_img[i-1])
        
        mat_kp=np.zeros([len(kp),(sizeDesc+2)])  
        for i in range(len(kp)):
            mat_kp[i][0]=kp[i].pt[0]
        for j in range(len(kp)):
            mat_kp[j][1]=kp[j].pt[1]
        mat_kp[:,2:(sizeDesc+2)]=desc
        
        np.savetxt(os.path.join(path_desc,filename+'.txt'),mat_kp,fmt='%f')
    
        rows, cols = desc.shape
        nb_kp_per_img = np.append(nb_kp_per_img, rows)
        all_desc = np.concatenate((all_desc, desc))

    
    [centroid_vec, val_dist] = Class.K_means(all_desc, nb_word, 5)
    sig = Class.Signature_img(all_desc, val_dist, nb_kp_per_img, nb_img, nb_word)
    
    sigTrain = sig[0:69]  
    sigTest = sig[70:100]
    res = np.zeros(0)
    
    for i in range(len(sigTest)):
        test = Class.KNN(sigTrain, ID, 5, sigTest[i], 0)
        res = np.append(res, test)
    
    return res, ID[70:100], list_path_img
    