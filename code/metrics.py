# -*- coding: utf-8 -*-
"""
Created on Wed May 20 14:46:42 2015

@author: Projet
"""

import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import XMLmodif
import XMLparser
import os
import TXTparser.py
from sphinx_doc import genere_doc
from sphinx_doc import configure_doc

def invlist(x):
    return 1./x
     

def metrics(p):
    
    ur"""
    
    This function calculate the score that the run files would get at the CLEF challenge. It needs the text file containing
    the observationID (the plant observation ID from which several pictures can be associated), the classID predicted (numerical taxonomical number used by Tela Botanica) and the
    rank (which represent the occurence of the observationID).
    
    
    The primary metric used to evaluate the submitted runs will be a score related to the rank of the correct species in the list of the retrieved species.
    Each plant observation test will be attributed with a score between 0 and 1 : of 1 if the 
    
    1st metric (score observation):
    .. math::
      S1=\frac{1}{U}\sum_{u=1}^{U}\frac{1}{P_{u}} \sum_{p=1}^{P_u}S_{u,p}
    
    U : number of users (who have at least one image in the test data)
    Pu : number of individual plants observed by the u-th user
    Su,p : score between 1 and 0 equals to the inverse of the rank of the correct species (for the p-th plant observed by the u-th user)    
    
    2nd metric (score image):
    
    ..math::
      S2=\frac{1}{U}\sum_{u=1}^{U}\frac{1}{P_{u}} \sum_{p=1}^{P_u}\frac{1}{N_{u,p}} \sum_{n=1}^{N_{n,p}}S_{u,p,n}
    
    
    U : number of users (who have at least one image in the test data)
    Pu : number of individual plants observed by the u-th user
    Nu,p : number of pictures taken from the p-th plant observed by the u-th user
    Su,p,n : score between 1 and 0 equals to the inverse of the rank of the correct species (for the n-th picture taken from the p-th plant observed by the u-th user)    
    
    
    Variables locales :
    :param Users: number of users who have at least one image in the test data
    :type Users:
    :param plants: number of plants observed by the u-th user
    :type plants:
    :param Sc: score between 1 and 0 equals to the inverse of the rank of the correct species (for the n-th picture taken from the p-th plant observed by the u-th user)
    :type Sc:
    :param N: number of pictures taken from the p-th plant observed by the u-th user
 
     
    :Example:
        
        
    >>> metrics()
    >>> 
     
    """
    #recup des données contenu dans le résultat du run
    #<test_image_name.jpg;ClassId;rank;score>    
    ranks=TXTparser()
    
    #inversion de chaque élément de ranks pour obtenir Sc
    Sc=map(invlist,ranks)
    
    
    #recuperer les donnees precalculees
    metx = open("metrics.xml", "r")    
    
    tree = ET.parse('metrics.xml')
    top = tree.getroot()
    
    k=0
    #revoir la structure du metrics.xml, preshot
    for authors in top.findall('author'):
        k+=1    
        users = author.find('count').text
        P[k] = author.find('plantID').text
        N[k] = author.find('nbpictures').text

    S1=0
    temp1=0
     #Primary metric
    #Average classification score
    for u in xrange(users):
        for p in xrange(len(P)):
            temp1+=Sc[u][p]
            
        S1+=temp1/P[u]
            #S1=1/users * sum(1/P[u] * sum(Sc[u][p],1),1)
    S1=S1/users
    
    temp1=0
    temp2=0
    S2=0
    #Secondary metric
    for u in xrange(users):
        for p in xrange(len(P)):
            for n in range(len(N)):
                temp1+=Sc[u][p][n]
            temp2+=temp1/N[u][p]
        S2+=temp2/P[u]
    S2=S2/users
                #S2+=1/P[u]+Sc[u][p]
                #S2=1/users * sum(1/P[u] * sum(1/N * sum(Sc[u][p],1),1),1)
    
    #enregistrement des scores
    sc=open("score.txt","wb")
    
    sc.write("primary metrics :"+S1+"  secondary metrics :"+S2)
    sc.close()
    metx.close()
    
    
    
    
def MetricsArgs(IBPath):
    
   
    """
    
    Cette fonction permet de récupérer tous les arguments nécessaires à l'évaluation. 
    Elle itère tous les fichiers xml à travers le dossiers de la banque d'image afin de 
    récupérer l'auteur et la plante de chaque image
    
    :param Users: number of users who have at least one image in the test data
    :type Users:

 
     
    :Example:
        
        
    >>> 
    >>> 
     
    """
    
    machin=zeros() 
    
    #creation xml pour stocker auteurs et nombre d'images puis calculer ?
    #something like 
#    <top>
#        <Auteur1>
#            <IDplant1>
#                <img1>
#                <img2>
#                ...
#            <IDplant2>
#                <img1>
#                <img2>                
#                ...
#        <Auteur2>
#            ...
    
#à faire tourner sur la banque d'image une fois
#puis plus qu'à compter, peut être réutilisé
#compatibilité avec runs partiels ?
    #iteration sur tous les fichiers XML de la banque d'images
    for i in os.listdir(IBPath):
        if i.endswith(".xml"): 
            
            [user, ObsID]=XMLparser(i+'.xml')
            XMLmodif(user, ObsID)            
#            while i<len(machin):
#               
#                if user==machin[i][j]:
#                    
#                    machin[i][j]+=1
#                    #recup l'auteur et l'id de la plante
#                    #incrementer le nombre d'image de l'auteur
#                    #incrementer le nombre d'image de ce l'id de cette plante
#                    #incrementer le nombre d'image de l'id de cette plante par cet auteur
#                    #stocker l'id ?
#                    #stocker l'auteur s'il n'est pas déjà enregistrer
#                    #stocker auteur et id et compter plus tard
#                    #a voir 
#                    #penser à recup le rank
#                    #passer par le module xml en c pour meilleures perfs ?         
#                else:            
            continue
    
    
    fo=open("run.txt","wb")

def metrxGraph (score1,score2):
    
   
    """
    
    This function draw a comparison between the score obtained and the scores from the oarticipants of the contest;
    
    :param Users: number of users who have at least one image in the test data
    :type Users:

 
     
    :Example:
        
        
    >>> 
    >>> 
     
    """
    

    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ## the data
    N = 5
    data=np.genfromtxt('CLEFresults2015')
    
    ## necessary variables
    ind = np.arange(N)                # the x locations for the groups
    width = 0.35                      # the width of the bars
    
    ## the bars
    rects1 = ax.bar(ind, Score1, width,
                    color='black')
    
    rects2 = ax.bar(ind+width, Score2, width,
                        color='red')
    
    # axes and labels
    ax.set_xlim(-width,len(ind)+width)
    ax.set_ylim(0,45)
    ax.set_ylabel('Scores')
    ax.set_title('Scores by teams')
    xTickMarks = [data[:][0]]
    ax.set_xticks(ind+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=45, fontsize=10)
    
    ## add a legend
    ax.legend( (rects1[0], rects2[0]), ('Score observation', 'Score image') )
    
    plt.show()    
        
    
    
    #other contestants scores :

        
        
genere_doc()    
    