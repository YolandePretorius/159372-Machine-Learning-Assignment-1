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
    
filenameIn = "bank-full.csv"
filenameOut = "bank-fullOut.csv"
df = 0
dict = {}
itemlist = []


# jobs = ["admin","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services"]
# marital_status = ["married","divorced","single"]
# education = ["unknown","secondary","primary","tertiary"]
# has_credit = ["yes","no"]

# print(jobs)
# read data into df(data frame)

def readDataFromFile(filename):
 
    # names = ['age','job','marital','education','default credit','housing','loan','contact','month','day_of_week','duration','campaign','pdays','previous','poutcome','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m',' nr.employed','subscribed']
    # df = pandas.read_csv(fname,names=names)
    # # print(df.head())
    # # print(df.isnull())
    # # print(pandas.isnull(df).sum())
    # np_df = df.values #converting panda data to numpy
    # # print(np_df)
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





def EncodeData(row,i): # change data to numerical values
    j = 0
  
    for item in row:
        if j in  dict:
            listDict = dict[j]
            if item in listDict:
                indextItem = listDict.index(item)
                newArrayData[i][j] = indextItem
            else: 
                listDict.append(item)
                indextItem = listDict.index(item)
                newArrayData[i][j] = indextItem
        else:
            dict[j] = []
            listDict = dict[j]
            listDict.append(item)
            indextItem = listDict.index(item)
            newArrayData[i][j] = indextItem
            print(newArrayData[i][j])
        j = j +1


def seperateData(ArrayData,i):
    
    for row in ArrayData:
        EncodeData(row,i)
        i = i+1
       
    print(newArrayData)
    
    
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
newArrayData = np.zeros(np.shape(numpy_df))
seperateData(numpy_df,i=0)
print(newArrayData[-20:])

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

