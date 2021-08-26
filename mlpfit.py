'''
Created on 14/08/2021

@author: yolan
This function determines the fitness of genomes which represents the weights of the multi layer percepetron.
The first weights are randomly generated and then senf through to the multi layer perceptron.
the fitness of the genomes determines if they will be used to create the next population. 

 
'''
import numpy as np

from _operator import index
from PartB import RunMLPB

testing_in,testing_tgt,train_in,train_tgt =RunMLPB.getDataMLP()

def mlpfit(pop):
    percentageAcuracyMax =0

    fitness = np.zeros(np.shape(pop)[0])
    index = 0
    # testing_in,testing_tgt,train_in,train_tgt =RunMLPB.getDataMLP()
    
    #print("Pop: ",sum(pop))
    print("Fitness function")
    for genome in pop:
        #print("sum 1",sum(genome))
        #print(np.shape(genome))
        weight1 = genome[:350] # for 10 hidden layers
        #print(np.shape(weight1))
        weight2 = genome[350:]
        #print(np.shape(weight2))
        weigth1Reshape = weight1.reshape(35,10)
        weigth2Reshape = weight2.reshape(11,2)
        
        
        
        percentAccuracy = RunMLPB.runMLP(testing_in,testing_tgt,train_in,train_tgt,weigth1Reshape,weigth2Reshape)
        print("Genome: ",sum(genome), " Weight1:",sum(weight1), " Weight2:",sum(weight2)," Percentage:", percentAccuracy)
        fitness[index] = percentAccuracy  # create a array with the fitness values per genome
        index +=1


    return(fitness)    
    
   
    
    