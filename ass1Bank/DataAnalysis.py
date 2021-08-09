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
    
filenameIn = "bank-full.csv"
# filenameTestIn = "bank.csv"
df = 0
dict = {}
itemlist = []
listNumericalData = [0,5,9,11,12,13,14]



# jobs = ["admin","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"]
# marital_status = ["married","divorced","single"]
# education = ["unknown","secondary","primary","tertiary"]
# has_credit = ["yes","no"]

# print(jobs)
# read data into df(data frame)

def readDataFromFile(filename):
 
    names = ['age','job','marital','education','default credit','housing','loan','contact','month','day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m',' nr.employed','subscribed']

    with open(filename, 'r') as f:
        np_df = list(csv.reader(f, delimiter=";"))
        
    np_df = np.array(np_df[1:])
    
    # print("last column", np_df[::,16:17])
    return np_df


def addDataToNUllArray(i,j,indextItem,item): # i: row number j: column number
    # print(item.dtype)
    if j in listNumericalData: # if the column is in the list representing the columns with numerical values in the array then add the numerical value other wise add the list indext number
 
        itemtype = float(item)
        newArrayData[i][j] = itemtype # store the numerical value (index) in the array to replace the string value
    else:
        newArrayData[i][j] = indextItem


def addDataToNUllArray2(i,j,indextItem,item): # i: row number j: column number
    # print(item.dtype)
        newArrayData[i][j] = indextItem

'''
Functions use a dictionary to encode strings of the array into numerical values. Key is the columns of the array [0 to 17] and values are a list of data items per row [0 to 45211] (strings) per column.
If item is already in the value list, then its not added
The string of data is converted to a numerical value using the index of the value list. 
The index of the list  is then saved in the position of the string forming a new array containing numerical values, newArrayData 
'''
def EncodeData(row,i): # i is the row value
    j = 0 # j is the column number used as the key in dictionary
    
    for item in row: 
        if j in  dict: # if the column is already a key 
            listDict = dict[j] # get the value (a list) for the key j 
            
            if item in listDict: # if item in list dont add 
                indextItem = listDict.index(item) # get index where item is in list. The index represents the numerical value of the string
                addDataToNUllArray(i,j,indextItem,item)
            else: 
                listDict.append(item)
                indextItem = listDict.index(item)
                addDataToNUllArray(i,j,indextItem,item)
        else:
            dict[j] = [] # create a list (will be value) for the key value pair
            listDict = dict[j]
            listDict.append(item)
            indextItem = listDict.index(item)
            addDataToNUllArray(i,j,indextItem,item)
        j = j +1 # move to next column in the row

        
'''
Send through the each row of the array containing data to be encoded as a numerical value
'''
def handle_non_numerical_data(ArrayData,i):

    for row in ArrayData:
            EncodeData(row,i)
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
    # newArrayData[:,:15] = newArrayData[:,:15]-newArrayData[:,:15].mean(axis=0)
    # newArrayData[:,:15] = newArrayData[:,:15]/newArrayData[:,:15].var(axis=0)
    newArrayData = (newArrayData - newArrayData.mean(axis=0))/newArrayData.var(axis=0)
    targets = (targets - targets.mean(axis=0))/targets.var(axis=0)
    # print(targets[10:])
    return newArrayData

def normalizeData2(newArrayData,column):
    minValue = np.min( newArrayData[:,column])
    maxValue = np.max( newArrayData[:,column])
    
    # print(min)
    # print(max)
    average = (maxValue -minValue)
    newArrayData[:,column] = newArrayData[:,column]-minValue
    newArrayData[:,column] = newArrayData[:,column]/average
    # targets = newArrayData[:,15]
   
    # newArrayData[:,column] = newArrayData[:,column]-newArrayData[:,column].mean(axis=0)
    # newArrayData[:,column] = newArrayData[:,column]/newArrayData[:,column].var(axis=0)
    # newArrayData = (newArrayData - newArrayData.mean(axis=1))/newArrayData.var(axis=0)
    # targets = (targets - targets.mean(axis=1))/targets.var(axis=0)
    # print(newArrayData[10:])
    return newArrayData

def ShuffleDataRandomly(newArrayData):
    # target = newArrayData[:,-1]
    order = np.arange(np.shape(newArrayData)[0])
    np.random.shuffle(order)
    newArrayData = newArrayData[order,:]
    # target = target[order]
    # target = target[order]
    # print(target[10:])
    return newArrayData


def seperateData70vs30(df,percentageTesting):
    testData, trainingData = BalanceSampling(df,percentageTesting)
    return testData, trainingData 
     

# def handle_non_numerical_data2(df):
#     columns = df.columns.values
#
#     for column in columns:
#         text_digit_vals = {}
#         def convert_to_int(val):
#             return text_digit_vals[val]
#
#         if df[column].dtype != np.int64 and df[column].dtype != np.float64:
#             column_contents = df[column].values.tolist()
#             unique_elements = set(column_contents)
#             x = 0
#             for unique in unique_elements:
#                 if unique not in text_digit_vals:
#                     text_digit_vals[unique] = x
#                     x+=1
#             df[column] = list(map(convert_to_int, df[column]))
#
#     return df        
    
def deleteColum(df,column):
    # print(np.shape(df))
    newData = np.delete(df,column, axis=1)
    # print(np.shape(newData))
    # print(deleteColumn7)
    return newData

def deleteRow(df,row):
    # print(np.shape(df))
    newData = np.delete(df,row, axis=0)
    # print(np.shape(newData))
    # print(deleteColumn7)
    return newData


def getRandomRow(DataArray):
    DataArray = ShuffleDataRandomly(DataArray)
    row = DataArray[:1]
    return row
    
def AddtoArray(NewArrayData,row,row_n):
    NewArrayData = np.insert(NewArrayData,row_n,[row],axis= 0)
    return NewArrayData


   
# create a data set with a 1:1 ratio of yes and no values
def BalanceSampling(DataArray, sizeArrayData):
    yesCounter = 0 # count the number yes target values is in the newArrayData
    noCounter = 0 #count the number no target values is in the newArrayData
    counter = 0
    
    ShuffleDataRandomly(DataArray)
    NewArrayData = np.array(DataArray[:1]) # add the first row of data to the array
    DataArray = deleteRow(DataArray,0) # Delete row to ensure that the data from original array is not added more than once.
    numberYes = round(sizeArrayData *0.5) #Divide the data 50% maximum number yes values to be added to the array
    numberNo = round(sizeArrayData *0.5) #Divide the data 50% no

    counter += 1
    valueYesOrNo = NewArrayData[:,-1]   
    if valueYesOrNo == 1:
        yesCounter+=1
    else:
        noCounter+=1
    
    while(counter <= sizeArrayData-1) and (counter < 10578): 
        row = getRandomRow(DataArray)
        valueYesOrNo = row[:,-1]  
        if valueYesOrNo == 1 and yesCounter <= numberYes:
            NewArrayData = AddtoArray(NewArrayData,row,counter)
            DataArray = deleteRow(DataArray,0)
            yesCounter+=1
            counter+=1
          
        if valueYesOrNo == 0 and noCounter < numberNo:
            NewArrayData = AddtoArray(NewArrayData,row,counter)
            DataArray = deleteRow(DataArray,0)
            noCounter+=1
            counter+=1     
 
    return  NewArrayData, DataArray  


# # def changeDataToNull(df):
# #     df[np.where(df[:] == "unknown")] = None         
# #     print(df)


# create a array with ones and diagonal values 
def KFoldcrossValidationData(data,numberfolds):
    numnerRows = (np.shape(data)[0]) # determine the number rows in the array
    numnerRowsPerfold = np.array_split(data,numberfolds, axis = 0) # split data in folds required
    diagonalOnesTable = np.eye(numberfolds, numberfolds)
    return numnerRowsPerfold, diagonalOnesTable

'''
---------------------------------main------------------------------------------------
'''
numpy_df = readDataFromFile(filenameIn)
print(np.shape(numpy_df))

numpy_df = ShuffleDataRandomly(numpy_df)


'''
count yes vs no 

'''
# number_YesOut1 = (np.where(numpy_df[:,-1] == "yes"))
# number_NoOut1  =  (np.where(numpy_df[:,-1] == "no"))
# print(np.shape(number_YesOut1)) 
# print(np.shape(number_NoOut1)) 



# unKnownNumber = np.where(numpy_df[:] == 'unknown')
# print("Unknown numbers", unKnownNumber)



newArrayData = np.zeros(np.shape(numpy_df)) # create zero array that will be used to hold the numerical values created for the strings

handle_non_numerical_data(numpy_df,i=0)

# print(newArrayData[:10])

newArrayData,validData = BalanceSampling(newArrayData,750)


# newArrayData = ageCategorization(newArrayData)

# print(np.shape(newArrayData))

# print(newArrayData)
#
# print(newArrayData[:15])

# newArrayData = ageCategorization(newArrayData)
# pl.plot(newArrayData[:,0],newArrayData[:,5],'ro')
# pl.show()




'''
normalizing data using min and max
'''
newArrayData = normalizeData2(newArrayData,0) # age
newArrayData = normalizeData2(newArrayData,1) #job
newArrayData = normalizeData2(newArrayData,2) #marital
newArrayData = normalizeData2(newArrayData,3) #education
newArrayData = normalizeData2(newArrayData,4) #default
newArrayData = normalizeData2(newArrayData,5) #balance
newArrayData = normalizeData2(newArrayData,6) #housing
newArrayData = normalizeData2(newArrayData,7) #loan
newArrayData = normalizeData2(newArrayData,8) #contact
newArrayData = normalizeData2(newArrayData,9) #day
newArrayData = normalizeData2(newArrayData,10) #month
newArrayData = normalizeData2(newArrayData,11) #duration
newArrayData = normalizeData2(newArrayData,12) #campaign
newArrayData = normalizeData2(newArrayData,13) #pdays
newArrayData = normalizeData2(newArrayData,14) #previous
newArrayData = normalizeData2(newArrayData,15)
# newArrayData = normalizeData2(newArrayData,16)


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
# newData = np.delete(newArrayData,10, axis=1)
newData = np.delete(newData,9, axis=1)
newData = np.delete(newData,8, axis=1)



sizeTestData = (np.shape(newArrayData)[0])*0.3
testData, trainingData = seperateData70vs30(newArrayData,sizeTestData)

folds, diagonalOnes = KFoldcrossValidationData(trainingData,3) # divide training data in folds

# distrubute__target_values_yes_No(training_data,testing_data)

# print(f"No. of training examples: {training_data.shape[0]}")
# print(f"No. of testing examples: {testing_data.shape[0]}")
# print("************************************************************")
# print(np.shape(trainingData))
# print("Training Data",trainingData[:])
# print("TestingData",testData[:])
# # valuesTraining = np.shape(training_data[0])
# print("************************************************************")

# print(np.shape(trainingData)[1])
# print(np.shape(testData)[1])

for x in diagonalOnes: 
    trainingDatafolds = []
    validationData = []
    for y in x:
        y =int(y)
        if y == 0: # if the diagonal value in the diagonalOnes data set is 0, it is  added to the training data dividing data as 1 part validation data and (n-1) part training data  
            if np.shape(trainingDatafolds)[0]== 0:
                trainingDatafolds = folds[y]
    
            else:
                # trainingDatafolds = np.append(folds[y])   
            
                trainingDatafolds = np.append(trainingDatafolds,folds[y],axis=0)
               
        else: # if the diagonal value in the diagonalOnes data set is 1, it is  set as the validation data 
            validationData  = folds[y]
            
            
    
    Data = np.shape(trainingData)[1]


    #train and test neural networks with different number of hidden neurons (i)
    results = np.array([(1,0),(2,0),(4,0),(6,0),(7,0),(8,0),(9,0),(10,0),(15,0),(20,0)])

    train_in = trainingDatafolds[::,:Data-1]
    train_tgt = trainingDatafolds[::,Data-1:Data]


    testing_in = testData[::,:Data-1]
    testing_tgt = testData[::,Data-1:Data]
    
    valid_in = validationData[::,:Data-1]
    valid_tgt = validationData[::,Data-1:Data]        
 
    
             
    for idx,i in np.ndenumerate(results[:,0]): 
        print("----- "+str(i))
        net = mlp.mlp(train_in,train_tgt,i,outtype = 'logistic')#different types of out puts: linear, logistic,softmax
        
        # net.mlptrain(train_in,train_tgt,i,100)
        # net.confmat(train_in,train_tgt)
        net.earlystopping(train_in,train_tgt,valid_in,valid_tgt,0.1) 
        percentageAccuracy = net.confmat(testing_in,testing_tgt)    
        # net.confmat(valid_in,valid_tgt)
        print("******************Percentage accuracy",percentageAccuracy)
        results[idx,1] = percentageAccuracy

#lets plot the first digit in the training set
pl.plot(results[:,0],results[:,1], label = "logistic")
pl.show()



# print(train_in,train_in)
# print(train_tgt)

# np.info(where)
# numpy_df = binary(df)
# print(numpy_df[:10])

# print(numpy_df[0])
# unknownvar1 = np.where(numpy_df[:]=="unknown") 
# print("Unknown1", unknownvar1) 
# numpy_df = deleteColum(numpy_df,15)
# numpy_df = deleteColum(numpy_df,8)
# print(numpy_df[:10])
#print(numpy_df[0])
# print("tail",numpy_df[-10:])

# changeDataToNull(numpy_df)
 
# unknownvar1 = np.where(numpy_df[:,19]=="unknown") 
# print("Unknown1",+ unknownvar1) 
# numpy_df = deleteColum(numpy_df,19)
# unknownvar2 = np.where(numpy_df[:,19]=="unknown")
# print(unknownvar2)  
# numpy_df = deleteColum(numpy_df,7) 
# # print(np.where(numpy_df[:,18]=="unknown"))

#######################Test If percepetron works###############################################

# anddata = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
# xordata = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
# p = mlp.mlp(anddata[:,0:2],anddata[:,2:3],2)
# p.mlptrain(anddata[:,0:2],anddata[:,2:3],0.25,1001)
# p.confmat(anddata[:,0:2],anddata[:,2:3])
# q = mlp.mlp(xordata[:,0:2],xordata[:,2:3],2)
# q.mlptrain(xordata[:,0:2],xordata[:,2:3],0.25,5001)
# q.confmat(xordata[:,0:2],xordata[:,2:3])

#############################################################