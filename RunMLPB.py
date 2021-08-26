'''
Created on 30/07/2021

@author: yolande Pretorius

Use this to train the MLP
'''

import numpy as np
import pylab as pl
from PartA import mlpPartA
import csv
from numpy import savetxt
import time
from scipy.linalg._solve_toeplitz import float64
from PartB import mlpPartB

    
filenameIn = "bank-full.csv"
# filenameTestIn = "bank.csv"
df = 0
dict = {}
itemlist = []
listNumericalData = [0,5,9,11,12,13,14]
lsitNonNumericalData = [1,2,3,6,7,8,9,10,15]
NumAfterDeletingColumns = [0,5,8,9,10]
testing_in = 0
testing_tgt = 0
train_in = 0
train_tgt = 0 

def readDataFromFile(filename):
 
    names = ['age','job','marital','education','default credit','housing','loan','contact','month','day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m',' nr.employed','subscribed']

    with open(filename, 'r') as f:
        np_df = list(csv.reader(f, delimiter=";"))
        
    np_df = np.array(np_df[1:])
    
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
# Find index of item in list  
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
            addDataToNUllArray(i,j,indextItem,item,newArrayData)
        j = j +1 # move to next column in the row
        
        
'''
Send through the each row of the array containing data to be encoded as a numerical value
'''
def handle_non_numerical_data(ArrayData,i, newArrayData):

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
    
# normalize numerical data 
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

def getTrainingData(df,rowsAdded):
    i = 0
    columns = (np.shape(df)[1])
    NewArrayData = np.empty((0,columns), float64)
    for i in range(df):
        if i in rowsAdded:
            pass
        else:
            NewArrayData = AddtoArray(NewArrayData,i)
    return  NewArrayData  
            
# Separate training 70% from testing data  30%
def seperateData70vs30(df,percentageTesting):
    testData,trainingData = BalanceSampling(df,percentageTesting)
    # trainingData = getTrainingData(df,rowsAdded) # remove rows added to the testing data
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
    


def AddtoArray(NewArrayData,row):
    if np.shape(NewArrayData)[0]== 0:
                NewArrayData = row
    else:
        NewArrayData = np.vstack((NewArrayData,row))
    return NewArrayData

def BalanceSampling2(DataArray, sizeArrayData):
    RowAddedToArray = []
    columns = (np.shape(DataArray)[1])
    numberYes = round(sizeArrayData *0.5)
    numberNo = round(sizeArrayData *0.5) 
    yesCounter = 0
    noCounter = 0
    DataArray = ShuffleDataRandomly(DataArray)
    YesValues = np.where(DataArray[:,-1] == 1)
    NoValues  = np.where(DataArray[:,-1] == 0)
    # NewArrayData =np.array([])
    NewArrayData = np.empty((0,columns), float64)
    if yesCounter <= numberYes:     
        for row_index in YesValues:
            for value  in row_index:            
                yesRow =  DataArray[value]
                RowAddedToArray.append(value)
                yesCounter+=1
                NewArrayData = AddtoArray(NewArrayData,yesRow)  
                       
    print("After Yes array size",np.shape(NewArrayData)) 
    
    for row_index in NoValues:
        for value  in row_index: 
            if noCounter <= numberNo:
                noRow =  DataArray[value]
                RowAddedToArray.append(value)
                noCounter+=1
                NewArrayData = AddtoArray(NewArrayData,noRow)         
    print("After NO array size",np.shape(NewArrayData)) 
    
    # delete the rows added to the new balanced array
    
    return NewArrayData,RowAddedToArray
                            

# create a data set with a 1:1 ratio of yes and no values
def BalanceSampling(DataArray, sizeArrayData):
    columns = (np.shape(DataArray)[1])
    yesCounter = 0 # count the number yes target values is in the newArrayData
    noCounter = 0 #count the number no target values is in the newArrayData
    counter = 0 # make sure the max amount of data rows is not exceeded  
    DataArray = ShuffleDataRandomly(DataArray)
    numberYes = round(sizeArrayData *0.5) #Divide the data 50% maximum number yes values to be added to the array
    numberNo = round(sizeArrayData *0.5) #Divide the data 50% no
    
    NewArrayData = np.empty((0,columns), float64)


    while(counter <= np.shape(DataArray)[0]): 
        row = DataArray[:1]
        valueYesOrNo = row[:,-1]  
        if valueYesOrNo == 1 and yesCounter <= numberYes:
            NewArrayData = AddtoArray(NewArrayData,row)
            DataArray = deleteRow(DataArray,0)
            yesCounter+=1
            
          
        if valueYesOrNo == 0 and noCounter <= numberNo:
            NewArrayData = AddtoArray(NewArrayData,row)
            DataArray = deleteRow(DataArray,0)
            noCounter+=1
        counter+=1
    print(np.shape(NewArrayData))   
    return  NewArrayData, DataArray  



def oneOfNEncodingByColumn(newArrayData):
    columns = np.shape(newArrayData)[1]
    rows = (np.shape(newArrayData)[0])
    
    OneOfNEncodingArray =  np.zeros((int(rows),1))
    for c in range(columns):
        
   
        if (c in NumAfterDeletingColumns):
            OneOfNEncodingArrayCol =  np.zeros((int(rows),1))
            OneOfNEncodingArrayCol =np.insert(OneOfNEncodingArrayCol, 1, newArrayData[:,c], axis=1)
            OneOfNEncodingArrayCol = np.delete(OneOfNEncodingArrayCol,0, axis=1)
            OneOfNEncodingArray = np.concatenate((OneOfNEncodingArray,OneOfNEncodingArrayCol), axis=1)

        else:
            r=0
            rownumber = 0
            maxValue = np.max(newArrayData[:,c])
            OneOfNEncodingArrayRow =  np.zeros((int(rows),int(maxValue)+1))
            for r in range(rows-1):              
                currentValue = newArrayData[r][c]
                OneOfNEncodingArrayRow[r,int(currentValue)] = 1
            rownumber+=1
            OneOfNEncodingArray = np.concatenate((OneOfNEncodingArray, OneOfNEncodingArrayRow), axis=1)
       
    OneOfNEncodingArray = np.delete(OneOfNEncodingArray,0, axis=1)  
      
    return OneOfNEncodingArray 
'''
---------------------------------main------------------------------------------------
'''
def getDataMLP():
    print("start get Data")
    
    
    
    numpy_df = readDataFromFile(filenameIn)
    
    
    '''
    Determine the yes to no ratio
    '''
    
    number_YesOut1 = sum(np.where(numpy_df[:,-1] == 'yes'))
    number_NoOut1  =  sum(np.where(numpy_df[:,-1] == 'no'))
    print("Number yes values",np.shape(number_YesOut1))
    print("Number no values",np.shape(number_NoOut1))
    # print(np.shape(numpy_df))
    
    # unKnownNumber = np.where(numpy_df[:] == 'unknown')
    # print("Unknown numbers", unKnownNumber)
    # newArrayData = np.zeros(np.shape(numpy_df)) #create zero array that will be used to hold the numerical values created for the strings
    
    print("create zero array")
    newArrayData = np.zeros(np.shape(numpy_df)) # create zero array that will be used to hold the numerical values created for the strings
    print("convert to numerical data")
    i= 0
    handle_non_numerical_data(numpy_df, i,newArrayData)
    print("balance data")
    
    
    newArrayDataBalanced,RowAddedToArray = BalanceSampling2(newArrayData, 10578)
    
    # newArrayDataBalanced,validData = BalanceSampling(newArrayData,10578)
    
    
    number_YesOut1 = sum(np.where(newArrayDataBalanced[:,-1] == 1))
    number_NoOut1  =  sum(np.where(newArrayDataBalanced[:,-1] == 0))
    print("Number yes values",np.shape(number_YesOut1))
    print("Number no values",np.shape(number_NoOut1))
    
    # pl.plot(newArrayDataBalanced[number_NoOut1,0],newArrayDataBalanced[number_NoOut1,11],'rx')
    # pl.plot(newArrayDataBalanced[number_YesOut1,0],newArrayDataBalanced[number_YesOut1,11],'go')
    #
    # pl.xlabel("Age")
    # pl.ylabel("duration of the call")
    
    
    '''
    normalizing numerical data using min and max
    '''
    print("Normalizing data")
    newArrayData = normalizeData2(newArrayDataBalanced,0) # age
    # newArrayData = normalizeData2(newArrayData,1) #job
    # newArrayData = normalizeData2(newArrayData,2) #marital
    # newArrayData = normalizeData2(newArrayData,3) #education
    # newArrayData = normalizeData2(newArrayData,4) #default
    newArrayData = normalizeData2(newArrayDataBalanced,5) #balance
    # newArrayData = normalizeData2(newArrayData,6) #housing
    # newArrayData = normalizeData2(newArrayData,7) #loan
    # newArrayData = normalizeData2(newArrayData,8) #contact
    # newArrayData = normalizeData2(newArrayData,9) #day remove
    # newArrayData = normalizeData2(newArrayData,10) #month  remove
    newArrayData = normalizeData2(newArrayDataBalanced,11) #duration
    newArrayData = normalizeData2(newArrayDataBalanced,12) #campaign
    newArrayData = normalizeData2(newArrayDataBalanced,13) #pdays
    newArrayData = normalizeData2(newArrayDataBalanced,14) #previous
    # newArrayData = normalizeData2(newArrayData,15) #poutcome
    # newArrayData = normalizeData2(newArrayData,16) # target
    
    
    
    
    # pl.plot(newArrayData[:,0],newArrayData[:,5],'ro')
    # pl.show()
    
    # '''
    # randomly shuffle data
    # '''
    # newArrayData = ShuffleDataRandomly(newArrayDataBalanced)
    
    
    '''
    remove column 8,9,10 and 11 
    '''
    print("Delete data")
    newData = np.delete(newArrayData,11, axis=1)
    newData = np.delete(newData,10, axis=1)
    newData = np.delete(newData,9, axis=1)
    newData = np.delete(newData,8, axis=1)
    
    print("Shape of data after deleting columns", np.shape(newData))
    
    
    
    
    '''
    Categorical data converted  into 1 of N encoding 
    
    '''
    print("Encode data")
    NewEncodedArray = oneOfNEncodingByColumn(newData)
    
    print("Shape of data after encoding data", np.shape(NewEncodedArray))
    
    storeDatainfile(NewEncodedArray)
    
    '''
    Seperate data into test and training set. 
    '''
    sizeTestData = round((np.shape(NewEncodedArray)[0])*0.3,0)
    testData, trainingData = seperateData70vs30(NewEncodedArray,sizeTestData)
    print("Test Data Shape",np.shape(testData))
    print("Training Data Shape",np.shape(trainingData))
    

    testDataCol = np.shape(testData)[1]
    testDataRow = np.shape(testData)[0]
    TrainingDataCol = np.shape(trainingData)[1]
    TrainingDataColDataRow = np.shape(trainingData)[0]
    
    
    testing_in = testData[::,:testDataCol-2]
    testing_tgt = testData[::,testDataCol-2:testDataCol]
    train_in = trainingData[::,:TrainingDataCol-2]
    train_tgt = trainingData[::,TrainingDataCol-2:TrainingDataCol]
    
    return  testing_in,testing_tgt,train_in,train_tgt


def runMLP(testing_in,testing_tgt,train_in,train_tgt,weight1,weight2):
    
    net = mlpPartB.mlp(train_in,train_tgt,10,weight1,weight2,outtype = 'softmax')#different types of out puts: linear, logistic,softmax
    # error = net.mlptrain(train_in,train_tgt,0.25,101)
    # errorEarlyStoppingError = net.earlystopping(train_in,train_tgt,train_in,train_tgt,5)
    percentageAccuracy = net.confmat(testing_in,testing_tgt)    
    return(percentageAccuracy)




# testing_in,testing_tgt,train_in,train_tgt = getDataMLP()




