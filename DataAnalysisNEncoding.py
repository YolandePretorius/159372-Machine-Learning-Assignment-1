'''
Created on 30/07/2021

@author: yolande Pretorius
'''

import numpy as np
import math
import pandas as pd
import pylab as pl
import mlp
from fileinput import filename
from numpy import genfromtxt, where, int0
import csv
from scipy.optimize import _group_columns
from matplotlib.pyplot import axis
import string

from numpy import asarray
from numpy import savetxt
# from ass1Bank.DataAnalysis import deleteColum
    
filenameIn = "bank.csv"
# filenameTestIn = "bank.csv"
df = 0
dict = {}
itemlist = []
listNumericalData = [0,5,9,11,12,13,14]
lsitNonNumericalData = [1,2,3,6,7,8,9,10,15]
NumAfterDeletingColumns = [0,5,8,9,10]



def readDataFromFile(filename):
 
    names = ['age','job','marital','education','default credit','housing','loan','contact','month','day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m',' nr.employed','subscribed']

    with open(filename, 'r') as f:
        np_df = list(csv.reader(f, delimiter=";"))
        
    np_df = np.array(np_df[1:])
    
    # print("last column", np_df[::,16:17])
    return np_df

def storeDatainfile(data):
    savetxt('bankDataAdjusted2.csv', data, delimiter=',')

def addDataToNUllArray(i,j,indextItem,item,newArrayData): # i: row number j: column number
    # print(item.dtype)
    if j in listNumericalData: # if the column is in the list representing the columns with numerical values in the array then add the numerical value other wise add the list indext number
        itemtype = float(item)
        newArrayData[i][j] = itemtype # store the numerical value (index) in the array to replace the string value
    else:
        newArrayData[i][j] = indextItem

def findIndext(listDict,item):
    i = 0
    for itemInList in listDict:
        if itemInList == item:
            return i
        else:
            i+=1

'''
Functions use a dictionary to encode strings of the array into numerical values. Key is the columns of the array [0 to 17] and values are a list of data items per row [0 to 45211] (strings) per column.
If item is already in the value list, then its not added
The string of data is converted to a numerical value using the index of the value list. 
The index of the list  is then saved in the position of the string forming a new array containing numerical values, newArrayData 
'''
def EncodeDataDictionary(row,i,newArrayData): # i is the row value
    j = 0 # j is the column number used as the key in dictionary
    
    for item in row: 
        if j in  dict: # if the column is already a key 
            listDict = dict[j] # get the value (a list) for the key j 
            
            if item in listDict: # if item in list dont add 
                indextItem = findIndext(listDict,item) # get index where item is in list. The index represents the numerical value of the string
                addDataToNUllArray(i,j,indextItem,item,newArrayData)
            else: 
                listDict.append(item)
                indextItem = findIndext(listDict,item)
                addDataToNUllArray(i,j,indextItem,item,newArrayData)
        else:
            dict[j] = [] # create a list (will be value) for the key value pair
            listDict = dict[j]
            listDict.append(item)
            indextItem = findIndext(listDict,item)
            # indextItem = np.where(listDict[j] == 'item')
            addDataToNUllArray(i,j,indextItem,item,newArrayData)
        j = j +1 # move to next column in the row
        
        
'''
Send through the each row of the array containing data to be encoded as a numerical value
'''
def handle_non_numerical_data(ArrayData,i,newArrayData):

    for row in ArrayData:
            EncodeDataDictionary(row, i,newArrayData)
            i = i+1 # row number
    
    
#Categorize the age data. <= 30 years is set to 1, 30-40 is set to 2, etc
def ageCategorization(newArrayData):
    newArrayData[np.where(newArrayData[:,0]<=30),0] = 1      
    newArrayData[np.where((newArrayData[:,0]>30) & (newArrayData[:,0]<=40)),0] = 2
    newArrayData[np.where((newArrayData[:,0]>40) & (newArrayData[:,0]<=50)),0] = 3
    newArrayData[np.where((newArrayData[:,0]>50) & (newArrayData[:,0]<=60)),0] = 4
    newArrayData[np.where(newArrayData[:,0]>60),0] = 5 

    return newArrayData
    
def normalizeData(newArrayData):
    targets = newArrayData[:,11]
    newArrayData = (newArrayData - newArrayData.mean(axis=0))/newArrayData.var(axis=0)
    targets = (targets - targets.mean(axis=0))/targets.var(axis=0)
    # print(targets[10:])
    return newArrayData

def normalizeData2(newArrayData,column):
    minValue = np.min( newArrayData[:,column])
    maxValue = np.max( newArrayData[:,column])
    
    average = (maxValue -minValue)
    if(average == 0):
        return newArrayData
    else:
        newArrayData[:,column] = newArrayData[:,column]-minValue
        newArrayData[:,column] = newArrayData[:,column]/average
        return newArrayData

def ShuffleDataRandomly(newArrayData):
    # target = newArrayData[:,-1]
    order = np.arange(np.shape(newArrayData)[0])
    np.random.shuffle(order)
    newArrayData = newArrayData[order,:]
    return newArrayData


def seperateData70vs30(df,percentageTesting):
    testData, trainingData = BalanceSampling(df,percentageTesting)
    return testData, trainingData 
     
def deleteColum(df,column):
    newData = np.delete(df,column, axis=1)
    return newData

def deleteRow(df,row):
    newData = np.delete(df,row, axis=0)
    return newData


def getRandomRow(DataArray):
    DataArray = ShuffleDataRandomly(DataArray)
    row = DataArray[:1]
    return row
    


def AddtoArray(NewArrayData,row,row_n):
    if np.shape(NewArrayData)[0]== 0:
                NewArrayData = row
    else:
        NewArrayData = np.append(NewArrayData,row,axis=0)
    return NewArrayData

   

# create a data set with a 1:1 ratio of yes and no values
def BalanceSampling(DataArray, sizeArrayData):
    yesCounter = 0 # count the number yes target values is in the newArrayData
    noCounter = 0 #count the number no target values is in the newArrayData
    counter = 0 # make sure the max amount of data rows is not exceeded
    
    ShuffleDataRandomly(DataArray)
    numberYes = round(sizeArrayData *0.5) #Divide the data 50% maximum number yes values to be added to the array
    numberNo = round(sizeArrayData *0.5) #Divide the data 50% no
    
    NewArrayData =[]


    while(counter <= np.shape(DataArray)[0]): 
        row = getRandomRow(DataArray)
        valueYesOrNo = row[:,-1]  
        if valueYesOrNo == 1 and yesCounter <= numberYes:
            NewArrayData = AddtoArray(NewArrayData,row,counter)
            DataArray = deleteRow(DataArray,0)
            yesCounter+=1
    
          
        if valueYesOrNo == 0 and noCounter < numberNo:
            NewArrayData = AddtoArray(NewArrayData,row,counter)
            DataArray = deleteRow(DataArray,0)
            noCounter+=1
         
        counter+=1
    return  NewArrayData, DataArray  


# create a array with ones and diagonal values 
def KFoldcrossValidationData(data,numberfolds):
    numnerRows = (np.shape(data)[0]) # determine the number rows in the array
    numnerRowsPerfold = np.array_split(data,numberfolds, axis = 0) # split data in folds required
    diagonalOnesTable = np.eye(numberfolds, numberfolds)
    return numnerRowsPerfold, diagonalOnesTable

def NEncoding(maxIndex, currentValue):
    
    ArrayZero = np.zeros(int(maxIndex)+1);
    ArrayZero[int(currentValue)] = 1
    return ArrayZero
    
def oneOfNEncodingByColumn(newArrayData):
    columns = np.shape(newArrayData)[1]
    rows = (np.shape(newArrayData)[0])
    
    # OneOfNEncodingArrayCol =  np.zeros((int(rows),1))
    OneOfNEncodingArray =  np.zeros((int(rows),1))
    # OneOfNEncodingtarget = np.zeros((int(rows),1))
    # ColumnArray = np.array([])
    # OneOfNEncodingArray = np.zeros(np.shape(numpy_df))
    for c in range(columns):
        
   
        if (c in NumAfterDeletingColumns):
            OneOfNEncodingArrayCol =  np.zeros((int(rows),1))
            OneOfNEncodingArrayCol =np.insert(OneOfNEncodingArrayCol, 1, newArrayData[:,c], axis=1)
            # OneOfNEncodingArrayCol = np.concatenate((OneOfNEncodingArrayCol, newArrayData[:,c]), axis=1)
            OneOfNEncodingArrayCol = np.delete(OneOfNEncodingArrayCol,0, axis=1)
            # print(np.shape(OneOfNEncodingArrayCol))
            # print(np.shape(OneOfNEncodingArray))
            OneOfNEncodingArray = np.concatenate((OneOfNEncodingArray,OneOfNEncodingArrayCol), axis=1)
            # OneOfNEncodingArrayCol =np.insert(OneOfNEncodingArrayCol, 1, newArrayData[:,c], axis=1)
            # OneOfNEncodingArray[:,c] = newArrayData[:,c]
            # print(np.shape(OneOfNEncodingArrayCol))
            # print(np.shape(OneOfNEncodingArray))
            # if np.shape(OneOfNEncodingArray)[0]== 0:
            #     OneOfNEncodingArray =  ColumnArray[:,0]
            #     shapeArrayA = OneOfNEncodingArray.shape
            #     OneOfNEncodingArray = shapeArrayA.reshape(-1,1)
            #     # OneOfNEncodingArray = np.transpose(OneOfNEncodingArray) 
            
        else:
            r=0
            rownumber = 0
            maxValue = np.max(newArrayData[:,c])
            OneOfNEncodingArrayRow =  np.zeros((int(rows),int(maxValue)+1))
            for r in range(rows-1):              
                currentValue = newArrayData[r][c]
                # nArray = NEncoding(maxValue, currentValue)
                OneOfNEncodingArrayRow[r,int(currentValue)] = 1
                # OneOfNEncodingArrayRow = np.insert(OneOfNEncodingArrayRow, rownumber, nArray, axis=0)
                # print(np.shape(OneOfNEncodingArrayRow))
                # OneOfNEncodingArrayRow[r:] = nArray
                # print()
            rownumber+=1
            # if (c == columns-1):
            #     OneOfNEncodingtarget = np.concatenate((OneOfNEncodingtarget, OneOfNEncodingArrayRow), axis=1)
            # else:
            OneOfNEncodingArray = np.concatenate((OneOfNEncodingArray, OneOfNEncodingArrayRow), axis=1)
            # print("SHAPE", np.shape(OneOfNEncodingArray))
            # print(OneOfNEncodingArray[:-20])        
    OneOfNEncodingArray = np.delete(OneOfNEncodingArray,0, axis=1)  
    # OneOfNEncodingtarget = np.delete(OneOfNEncodingtarget,0, axis=1)       
    return OneOfNEncodingArray 
'''
---------------------------------main------------------------------------------------
'''

    
def runMLP():
    print("start runMLP")
    numpy_df = readDataFromFile(filenameIn)
    
    # print(np.shape(numpy_df))
    
    # numpy_df = ShuffleDataRandomly(numpy_df)
    
    # unKnownNumber = np.where(numpy_df[:] == 'unknown')
    # print("Unknown numbers", unKnownNumber)
    # newArrayData = np.zeros(np.shape(numpy_df)) #create zero array that will be used to hold the numerical values created for the strings
    
    # newArrayDataBalanced,validData = BalanceSampling(numpy_df,1000)
    
    newArrayData = np.zeros(np.shape(numpy_df)) # create zero array that will be used to hold the numerical values created for the strings
    i=0
    handle_non_numerical_data(numpy_df,i,newArrayData)
    
    newArrayDataBalanced,validData = BalanceSampling(newArrayData,10578)
    
    
    # '''
    # count yes vs no 
    #
    # '''
    # number_YesOut1 = (np.where(newArrayDataBalanced[:,-1] == 1))
    # number_NoOut1  =  (np.where(newArrayDataBalanced[:,-1] == 0))
    # pl.plot(newArrayDataBalanced[number_NoOut1,0],newArrayDataBalanced[number_NoOut1,11],'rx')
    # pl.plot(newArrayDataBalanced[number_YesOut1,0],newArrayDataBalanced[number_YesOut1,11],'go')
    #
    # pl.xlabel("Age")
    # pl.ylabel("duration of the call")
    
    # print(np.shape(number_YesOut1)) 
    # print(np.shape(number_NoOut1))
    
    # newArrayData,validData = BalanceSampling(newArrayData,1000)
    
    # OneOfNEncodingArray = np.zeros(np.shape(numpy_df))
    #
    # OneOfNEncodingArray = np.zeros(np.shape(numpy_df))
    
    '''
    Seperate data in a ballanced smaller data set
    
    '''
    # newArrayData,validData = BalanceSampling(newArrayData,500)
    
    
    
    
    
    '''
    normalizing data using min and max
    '''
    newArrayData = normalizeData2(newArrayData,0) # age
    # newArrayData = normalizeData2(newArrayData,1) #job
    # newArrayData = normalizeData2(newArrayData,2) #marital
    # newArrayData = normalizeData2(newArrayData,3) #education
    # newArrayData = normalizeData2(newArrayData,4) #default
    newArrayData = normalizeData2(newArrayData,5) #balance
    # newArrayData = normalizeData2(newArrayData,6) #housing
    # newArrayData = normalizeData2(newArrayData,7) #loan
    # newArrayData = normalizeData2(newArrayData,8) #contact
    # newArrayData = normalizeData2(newArrayData,9) #day
    # newArrayData = normalizeData2(newArrayData,10) #month
    newArrayData = normalizeData2(newArrayData,11) #duration
    newArrayData = normalizeData2(newArrayData,12) #campaign
    newArrayData = normalizeData2(newArrayData,13) #pdays
    newArrayData = normalizeData2(newArrayData,14) #previous
    # newArrayData = normalizeData2(newArrayData,15) #poutcome
    # newArrayData = normalizeData2(newArrayData,16) # target
    
    
    # pl.plot(newArrayData[:,0],newArrayData[:,5],'ro')
    # pl.show()
    
    '''
    randomly shuffle data
    '''
    newArrayData = ShuffleDataRandomly(newArrayData)
    
    '''
    remove column 8 and column 10 
    '''
    # print(np.shape(newArrayData))
    newData = np.delete(newArrayData,11, axis=1)
    newData = np.delete(newData,10, axis=1)
    newData = np.delete(newData,9, axis=1)
    newData = np.delete(newData,8, axis=1)
    
    # print(np.shape(newData))
    
    
    
    NewEncodedArray = oneOfNEncodingByColumn(newData)
    # deleteColum(NewEncodedArray,0)
    # deleteColum(NewEncodedArray,1)
    
    
    storeDatainfile(NewEncodedArray)
    
    
    sizeTestData = (np.shape(NewEncodedArray)[0])*0.3
    testData, trainingData = seperateData70vs30(NewEncodedArray,sizeTestData)
    
    # folds, diagonalOnes = KFoldcrossValidationData(trainingData,3) # divide training data in folds
    
    # use different combinations of k-fold cross validation values 
    ############################################################################
    results = np.array([(15,0)])
    
    testDataCol = np.shape(testData)[1]
    testDataRow = np.shape(testData)[0]
    TrainingDataCol = np.shape(trainingData)[1]
    TrainingDataColDataRow = np.shape(trainingData)[0]
    
    
    testing_in = testData[::,:testDataCol-2]
    testing_tgt = testData[::,testDataCol-2:testDataCol]
    train_in = trainingData[::,:TrainingDataCol-2]
    traint_gt = trainingData[::,TrainingDataCol-2:TrainingDataCol]
    
    
    for idx,i in np.ndenumerate(results[:,0]):
         
            # print("----- "+str(i))
            # print(np.shape(net.weights1))
            # print(np.shape(net.weights2))
            # weights1 = 0
            # weights2 = 0
            #
    
            # nin = np.shape(train_in)[1]
            # nout = np.shape(traint_gt)[1]
            #
            # weights1 = (np.random.rand(nin+1,i)-0.5)*2/np.sqrt(nin)
            # weights2 = (np.random.rand(i+1,nout)-0.5)*2/np.sqrt(i)
    
            net = mlp.mlp(train_in,traint_gt,i,outtype = 'softmax')#different types of out puts: linear, logistic,softmax
            # weights1,weights2 = net.mlptrain(train_in,train_tgt,0.25,101)
            # print("weights 1",weights1)
            # print("weights 2",weights2)
            error = net.mlptrain(train_in,traint_gt,0.25,101)
            errorEarlyStoppingError = net.earlystopping(train_in,traint_gt,train_in,traint_gt,0.1) 
            percentageAccuracy = net.confmat(testing_in,testing_tgt)    
            results[idx,1] = percentageAccuracy
            # weights1,weights2 = net.mlpfwd(inputs)
            # weights2 = net.weights2
            # for item in weights1:
            #     print(item)
            # print(np.shape(net.weights1))
            # print(np.shape(net.weights2))
    
    
    pl.plot(results[:,0],results[:,1])
    pl.show()


# runMLP()




############################################################################


# for x in diagonalOnes: 
#     trainingDatafolds = []
#     validationData = []
#     for y in x:
#         y =int(y)
#         if y == 0: # if the diagonal value in the diagonalOnes data set is 0, it is  added to the training data dividing data as 1 part validation data and (n-1) part training data  
#             if np.shape(trainingDatafolds)[0]== 0:
#                 trainingDatafolds = folds[y]
#
#             else:
#                 # trainingDatafolds = np.append(folds[y])   
#
#                 trainingDatafolds = np.append(trainingDatafolds,folds[y],axis=0)
#
#         else: # if the diagonal value in the diagonalOnes data set is 1, it is  set as the validation data 
#             validationData  = folds[y]
#
#
#
#     TrainingDataCol = np.shape(trainingDatafolds)[1]
#     TrainingDataColDataRow = np.shape(trainingDatafolds)[0]
#
#     validationDataCol = np.shape(validationData)[1]
#     validationDataRow = np.shape(validationData)[0]
#
#     testDataCol = np.shape(testData)[1]
#     testDataRow = np.shape(testData)[0]
#
#     #train and test neural networks with different number of hidden neurons (i)
#     results = np.array([(1,0),(2,0),(4,0),(6,0),(7,0),(8,0),(9,0),(10,0),(15,0),(20,0)])
#
#     train_in = trainingDatafolds[::,:TrainingDataCol-2]
#     traint_gt = trainingDatafolds[::,TrainingDataCol-2:TrainingDataCol]
#
#     # i = 0
#     # train_tgtZero = np.zeros((TrainingDataColDataRow,2))
#     # for i in range(TrainingDataColDataRow):
#     #     col = int(train_tgt[i][0])
#     #     train_tgtZero[i][col] = 1
#     #
#     # print(train_tgtZero[-20:])
#     #
#
#     testing_in = testData[::,:testDataCol-2]
#     testing_tgt = testData[::,testDataCol-2:testDataCol]
#
#     # test_tgtZero = np.zeros((testDataRow,2))
#     # for i in range(testDataRow):
#     #     col = int(testing_tgt[i][0])
#     #     test_tgtZero[i][col] = 1
#     #
#
#     valid_in = validationData[::,:validationDataCol-2]
#     valid_tgt = validationData[::,validationDataCol-2:validationDataCol] 
#
#     # valid_tgtZero = np.zeros((validationDataRow,2))
#     # for i in range(validationDataRow):
#     #     col = int(valid_tgt[i][0])
#     #     valid_tgtZero[i][col] = 1      
#     #
#
#     print('--------------------------------------------------')
#
#     for idx,i in np.ndenumerate(results[:,0]): 
#         print("----- "+str(i))
#         # print(np.shape(net.weights1))
#         # print(np.shape(net.weights2))
#         # weights1 = 0
#         # weights2 = 0
#         #
#
#         nin = np.shape(train_in)[1]
#         nout = np.shape(traint_gt)[1]
#
#         weights1 = (np.random.rand(nin+1,i)-0.5)*2/np.sqrt(nin)
#         weights2 = (np.random.rand(i+1,nout)-0.5)*2/np.sqrt(i)
#
#         net = mlp.mlp(train_in,traint_gt,i,outtype = 'softmax',weights1,weights2)#different types of out puts: linear, logistic,softmax
#         # weights1,weights2 = net.mlptrain(train_in,train_tgt,0.25,101)
#         # print("weights 1",weights1)
#         # print("weights 2",weights2)
#         error = net.mlptrain(train_in,traint_gt,0.25,101)
#         errorEarlyStoppingError = net.earlystopping(train_in,traint_gt,valid_in,valid_tgt,0.1) 
#         percentageAccuracy = net.confmat(testing_in,testing_tgt)    
#         results[idx,1] = percentageAccuracy
#         # weights1,weights2 = net.mlpfwd(inputs)
#         # weights2 = net.weights2
#         # for item in weights1:
#         #     print(item)
#         print(np.shape(net.weights1))
#         print(np.shape(net.weights2))
#
#
#         pl.plot(results[:,0],results[:,1])
#         pl.show()
#

#
# numpy_df = readDataFromFile(filenameIn)

