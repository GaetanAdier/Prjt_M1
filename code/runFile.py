# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:47:33 2015

@author: Projet
"""

def runFile(observationId,classId,rank,score):
    """
    
    Cette fonction permet de creer un fichier texte contenant les résultats du run sous la forme standardisée pour lifeCLEF.
    C'est-à-dire : <filename;ClassId;rank;score>
     `voir demandes<http://www.imageclef.org/lifeclef/2015/plant/>`_
    penser à renommer le fichier en "teamname_runX"
    
    :param observationId: the plant observation ID from which several pictures can be associated 
    :type observationId: int
    :param classId: the class number ID that must be used as ground-truth. It is a numerical taxonomical number used by Tela Botanica.
    :type classId: int
    :param rank: <rank> is the ranking of a given species for a given test ObservationId 
    :type rank: int
    :param score: <Score> is a confidence score of a prediction item (the lower the score the lower the confidence)
    :type score: float

     
    :Example:
        
        
    >>> 
    >>> 
     
    """
    
    #<test_image_name.jpg;ClassId;rank;score>
    line=str(observationId)+';'+str(classId)+';'+str(rank)+';'+str(score)+'\n'
    
    with open('run.txt', 'a') as file:
        file.write(line)
    
        
        
        
    