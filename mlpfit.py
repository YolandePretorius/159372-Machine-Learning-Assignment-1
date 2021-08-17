'''
Created on 14/08/2021

@author: yolan
This function determines the fitness of genomes which represents the weights of the multi layer percepetron.
The first weights are randomly generated and then senf through to the multi layer perceptron.
the fitness of the genomes determines if they will be used to create the next population. 

 
'''
import numpy as np
import DataAnalysisNEncoding
from _operator import index

def mlpfit(pop):
    percentageAcuracyMax =0

    fitness = np.zeros(np.shape(pop)[0])
    index = 0
    for genome in pop:
        
        weight1 = genome[:350] # for 10 hidden layers
        weight2 = genome[350:372]
        weigth1Reshape = weight1.reshape(35,10)
        weigth2Reshape = weight2.reshape(11,2)
        
        
        # for one hidden layer
        # weight1 = genome[:35] # for 1 hidden layers
        # weight2 = genome[35:39]
        # weigth1Reshape = weight1.reshape(35,1)
        # weigth2Reshape = weight2.reshape(2,2)
        #

        
        testing_in,testing_tgt,train_in,train_tgt =DataAnalysisNEncoding.getData()
        percentAccuracy = DataAnalysisNEncoding.runMLP(testing_in,testing_tgt,train_in,train_tgt,weigth1Reshape,weigth2Reshape)
        print(percentAccuracy)
        fitness[index] = percentAccuracy  # create a array with the fitness values per genome
        index +=1


    return(fitness)    
    
   
    
    