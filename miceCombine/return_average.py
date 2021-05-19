import os
import pandas as pd
import statistics
from collections import Counter 
from miceCombine.return_address import return_address

def return_mice_indexfile_address(miss_method, index_case, index_miss, index_file):
    dataMainPath = os.path.join(os.getcwd(), "data_stored", "data_mice_store")
    methodAddress = return_address(dataMainPath, miss_method, "method")
    caseAddress = return_address(methodAddress, str(index_case), "case")
    miceAddress = return_address(caseAddress, str(index_miss), "miss")
    miceFileAddress = return_address(miceAddress, str(index_file), "miss_file")
    miceFilesAddress = [os.path.join(miceFileAddress, i) for i in os.listdir(miceFileAddress) if i != '.DS_Store']
    miceFilesList = [pd.read_csv(i) for i in miceFilesAddress]
            
    return miceFilesList

def return_mice_indexfile_storeadd(miss_method, index_case, index_miss, index_file):
    dataMainPath = os.path.join(os.getcwd(), "data_stored", "data_mice")
    if not os.path.exists(dataMainPath):
        os.mkdir(dataMainPath)
    methodPath = os.path.join(dataMainPath, miss_method)
    if not os.path.exists(methodPath):
        os.mkdir(methodPath)
    casePath = os.path.join(methodPath, 'Case{0}'.format(index_case))
    if not os.path.exists(casePath):
        os.mkdir(casePath)
    missPath = os.path.join(casePath, 'miss{0}'.format(index_miss))
    if not os.path.exists(missPath):
        os.mkdir(missPath)
    fileAddress = os.path.join(missPath, '{0}.csv'.format(index_file))
    return fileAddress
    

def return_average_mice(miss_method, index_case, index_miss, index_file):
    dfList = return_mice_indexfile_address(miss_method, index_case, index_miss, index_file)
    dfres = pd.DataFrame()
    length = len(dfList)
    shape = dfList[0].shape
    columnNames = dfList[0].columns
    for index, i in enumerate(columnNames):
        consider = [df[i].to_list() for df in dfList]
        if i.split('_')[0] == "con":
            List = [statistics.mean([consider[k][j] for k in range(length)]) for j in range(shape[0])]
        else:
            List = [Counter([consider[k][j] for k in range(length)]).most_common()[0][0] for j in range(shape[0])]
        dfres[i] = List
    
    storeAddress = return_mice_indexfile_storeadd(miss_method, index_case, index_miss, index_file)
    dfres.to_csv(storeAddress, index = False)

    




