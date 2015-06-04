# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:24:13 2015

@author: Projet
"""
#import numpy as np

def TXTparser():
    
    
    with open('fakeRun.txt', 'r') as f:
        data = f.readlines()
    
    rank=[]
    #i=0
    for line in data:
        words = line.split(";")
        rank.append(int(words[2]))
        #print rank[i]
        #i+=1
        
    #print rank
    
    #autre solution :
    #recupérer tout dans un array    
    #data=np.loadtxt('fakerun.txt',delimiter=';')
    
    
    return rank