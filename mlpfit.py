'''
Created on 14/08/2021

@author: yolan
'''
import numpy as np
import DataAnalysisNEncoding

def mlpfit(pop):
    percentageAcuracyMax =0
  
    for genome in pop:
        '''
        weight1 = genome[:350] # for 10 hidden layers
        weight2 = genome[350:372]
        weigth1Reshape = weight1.reshape(35,10)
        weigth2Reshape = weight2.reshape(11,2)
        '''
        weight1 = genome[:35] # for 1 hidden layers
        weight2 = genome[35:39]
        weigth1Reshape = weight1.reshape(35,1)
        weigth2Reshape = weight2.reshape(2,2)
        
        
        testing_in,testing_tgt,train_in,train_tgt =DataAnalysisNEncoding.getData()
        percentAccuracy = DataAnalysisNEncoding.runMLP(testing_in,testing_tgt,train_in,train_tgt,weigth1Reshape,weigth2Reshape)
        # error = DataAnalysisNEncoding.runMLP(weigth1Reshape, weigth2Reshape)
        if (percentAccuracy  > percentageAcuracyMax):
            percentageAcuracyMax = percentAccuracy
            maxgenome = genome
        print(np.shape(weight1))
        print(np.shape(weight2))
        print(np.shape(weigth1Reshape))
        print(np.shape(weigth2Reshape))
        # print(np.shape(weights1))
        # print(np.shape(weights2))
        print(np.shape(pop))
    print(maxgenome)
    return(maxgenome)    
    
   
    
    