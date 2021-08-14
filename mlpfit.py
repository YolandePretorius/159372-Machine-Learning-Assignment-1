'''
Created on 14/08/2021

@author: yolan
'''
import numpy as np
import DataAnalysisNEncoding

def mlpfit(pop):
    for genome in pop:
        weight1 = genome[:525]
        weight2 = genome[525:557]
        weigth1Reshape = weight1.reshape(35,15)
        weigth2Reshape = weight2.reshape(16,2)
        print(np.shape(weight1))
        print(np.shape(weight2))
        print(np.shape(weigth1Reshape))
        print(np.shape(weigth2Reshape))
        # print(np.shape(weights1))
        # print(np.shape(weights2))
        print(np.shape(pop))
    
   
    
    