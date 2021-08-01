'''
Created on 30/07/2021

@author: yolande Pretorius
'''

import numpy as np
import mlp
import pandas
from fileinput import filename
from numpy import genfromtxt, where
import csv
import PrepData
from scipy.optimize import _group_columns
from scipy.linalg._solve_toeplitz import float64
    
filenameIn = "bank-full.csv"
filenameOut = "bank-fullOut.csv"
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
    # df = pandas.read_csv(filename)
    # print(df.head())
    # print(df.isnull())
    # print(pandas.isnull(df).sum())
    # np_df = df.values #converting panda data to numpy
    # print(np_df)
    with open(filename, 'r') as f:
        np_df = list(csv.reader(f, delimiter=";"))
    
    np_df = np.array(np_df[1:])
    
    print(np.shape(np_df))
    return np_df


# def binary(df):
#     yesString = 'yes'
#     noString = 'no'
#
#     for s in df:
#         if s.find(yesString)>-1:
#
#
#     # newdf = np.where(df =='no',1,0)
#     # newdf= df[np.where(df[:,17] =='yes'),17] = 1
#
#     # df = np.where(df == 'no', 0, df)
#     # # df = np.where(df == 'yes', 1, df)
#     # x = np.where(df == 'yes')
#     # df[np.where(df[:]=='yes'),4] = 1
#     print(newdf)
#
#     return newdf
#     # np.where(x < 5, x, -1)
#     # df[np.where(df[:,column]<10),column] =0   
#     # df[np.where(df[:]=='no'),4] = 0
#     # df[np.where(df[:]=='yes'),4] = 1
#     #

def addDataToNUllArray(i,j,indextItem,item): # i: row number j: column number
    # print(item.dtype)
    if j in listNumericalData: # if the column is in the list representing the columns with numerical values in the array then add the numerical value other wise add the list indext number
        itemtype = float64(item)
        newArrayData[i][j] = itemtype # store the numerical value (index) in the array to replace the string value
    else:
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
            print(newArrayData[i][j])
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
    newArrayData[np.where(newArrayData[:,0]>60),7] = 5 
    
def normalizeData(newArrayData):
    targets = newArrayData[:,15]
    # newArrayData[:,:15] = newArrayData[:,:15]-newArrayData[:,:15].mean(axis=0)
    # newArrayData[:,:15] = newArrayData[:,:15]/newArrayData[:,:15].var(axis=0)
    newArrayData = (newArrayData - newArrayData.mean(axis=0))/newArrayData.var(axis=0)
    targets = (targets - targets.mean(axis=0))/targets.var(axis=0)
    print(targets[10:])
    return newArrayData, targets

def ShuffleDataRandomly(newArrayData):
    target = newArrayData[:,16]
    order = np.arange(np.shape(newArrayData)[0])
    np.random.shuffle(order)
    newArrayData = newArrayData[order,:]
    target = target[order]
    # target = target[order]
    # print(target[10:])
    return newArrayData

       
def handle_non_numerical_data2(df):
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            df[column] = list(map(convert_to_int, df[column]))
            
    return df        
    
# def deleteColum(df,column):
#     print(np.shape(df))
#     newData = np.delete(df,column, axis=1)
#     print(np.shape(newData))
#     # print(deleteColumn7)
#     return newData
    
# def changeDataToNull(df):
#     df[np.where(df[:] == "unknown")] = None         
#     print(df)
#

# PrepData.preprocessBank(filename, filenameOut)
# print(filenameOut)

numpy_df = readDataFromFile(filenameIn)
# print(numpy_df)
newArrayData = np.zeros(np.shape(numpy_df)) # create zero array that will be used to hold the numerical values created for the strings
# df = handle_non_numerical_data(numpy_df)
handle_non_numerical_data(numpy_df,i=0)

# print(newArrayData[:10])
ageCategorization(newArrayData)

print(np.shape(df))

print(newArrayData[:10])

newArrayData, targets = normalizeData(newArrayData)
print("*********************************************")
print(newArrayData[:10])
shuffledArray = ShuffleDataRandomly(newArrayData)
print("*********************************************")
print(shuffledArray[:10])




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

