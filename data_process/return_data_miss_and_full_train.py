import os
import numpy as np
import pandas as pd
from data_process.return_address import return_address

def return_data_file(miss_method, index_case, index_miss, index_file):
    fullMainPath = os.path.join(os.getcwd(), 'data_stored', 'data')
    fullFileAddress = return_address(fullMainPath, str(index_case), "full_file")
    
    missMainPath = os.path.join(os.getcwd(), 'data_stored', 'data_miss')
    methodAddress = return_address(missMainPath, miss_method, "method")
    caseAddress = return_address(methodAddress, str(index_case), "case")
    missAddress = return_address(caseAddress, str(index_miss), "miss")
    missFileAddress = return_address(missAddress, str(index_file), "miss_file")
    df_miss = pd.read_csv(missFileAddress)
    df_full = pd.read_csv(fullFileAddress)
            
    return df_miss, df_full

def return_split_con_cat_column(df_miss, df_full):
    columns = df_miss.columns
    columns_split = [i.split('_') for i in columns]
    con = []
    cat = []
    for index, i in enumerate(columns_split):
        if i[0][0:3] == 'con':
            con.append(columns[index])
        elif i[0][0:3] == 'cat':
            cat.append(columns[index])
        else:
            continue
    
    return con, cat

def return_dictionary_column(column_unique):
    dict_res = {}
    dict_rev = {}
    for index, i in enumerate(column_unique):
        dict_res[i] = index
        dict_rev[index] = i
    
    return dict_res, dict_rev 

def return_encode_column(column_m, dictionary, mode):
    column_encode = []
    
    if mode == 'one_hot':
        for index, i in enumerate(column_m):
            if i == np.nan or str(i) == 'nan':
                column_encode.append([np.nan for j in range(len(dictionary))])
            else:
                column_encode.append(list(np.eye(len(dictionary))[dictionary[i]]))
                
        array_encode = np.array(column_encode, dtype = np.float32)
    
    elif mode == 'embedding':
        for index, i in enumerate(column_m):
            if i == np.nan or str(i) == 'nan':
                column_encode.append([np.nan for j in range(len(dictionary))])
            else:
                column_encode.append(list(np.eye(len(dictionary))[dictionary[i]] * 2. - 1.))
                
        array_encode = np.array(column_encode, dtype = np.float32)
        
    return array_encode
    
    
def return_normalized_column(column_m, column_i, mode):
    
    if mode == 'one_hot':
        max_m = np.nanmax(np.array(column_m))
        min_m = np.nanmin(np.array(column_m))
        max_i = np.max(np.array(column_i))
        min_i = np.min(np.array(column_i))
        if min_m != max_m:
            column_m_normalize = [(i - min_m)/(max_m - min_m) for i in column_m]
            column_m_normalize = [i for i in column_m_normalize]
        else:
            column_m_normalize = [i * 0. for i in column_m]
    elif mode == 'embedding':
        max_m = np.nanmax(np.array(column_m))
        min_m = np.nanmin(np.array(column_m))
        max_i = np.max(np.array(column_i))
        min_i = np.min(np.array(column_i))
        if min_m != max_m:
            column_m_normalize = [(i - min_m)/(max_m - min_m) for i in column_m]
            column_m_normalize = [i * 2. - 1. for i in column_m_normalize]
        else:
            column_m_normalize = [0. for i in column_m]
    
    array_m_normalized = np.array(column_m_normalize, np.float32).reshape(len(column_m_normalize), 1)
    
    return array_m_normalized, max_m, min_m, max_i, min_i

def return_data_miss_and_full_train(miss_method, index_case, index_miss, index_file, mode = 'one_hot'):
    #print("Now, Data Frames for Case {0} with full version, missing {1:.2%} are being prepared...".format(index_case, float(index_miss/100)))
    df_miss, df_full = return_data_file(miss_method, index_case, index_miss, index_file)
    con, cat = return_split_con_cat_column(df_miss = df_miss, df_full = df_full)
    #print(con, cat)
    # Define the labels, location below:
    labels = []
    labels_ori = []
    locations = []
    attach = []
    
    # Start encode the cat variables:
    #print("Now, the preprocessing of data have already started, with mode: " + mode)
    for index, i in enumerate(df_full.columns):
        if i in cat:
            column_m = df_miss[i].to_list()
            column_i = df_full[i].to_list()
            columns_m_unique = list(set([i for i in column_m if str(i) != 'nan']))
            columns_i_unique = list(set([i for i in column_i]))
            dict_res_column, dict_rev_column = return_dictionary_column(column_unique = columns_m_unique)
            dictori_res_column, dictori_rev_column = return_dictionary_column(column_unique = columns_i_unique)
            column_encode = return_encode_column(column_m = column_m, dictionary = dict_res_column, mode = mode)
            locations.append(column_encode.shape[1])
            labels.append(['cat', [dict_res_column, dict_rev_column]])
            labels_ori.append(['cat', [dictori_res_column, dictori_rev_column]])
            attach.append(['cat', columns_i_unique]) 
        elif i in con:
            column_m = df_miss[i].to_list()
            column_i = df_full[i].to_list()
            column_encode, max_m, min_m, max_i, min_i = return_normalized_column(column_m = column_m, column_i = column_i, mode = mode)
            locations.append(column_encode.shape[1])
            labels.append(['con', [max_m, min_m]])
            labels_ori.append(['con', [max_i, min_i]])
            attach.append(['con', [max_m, min_m]])
            
        array_encode_column = np.array(column_encode, np.float32)

        if index == 0:
            array_result = array_encode_column
        else:
            array_result = np.concatenate((array_result, array_encode_column), axis = 1)
            locations[-1] = locations[-1] + locations[-2]
    #print("Now, the preprocessing of data had been finished.\n")   
    return array_result, df_full.columns, locations, labels, df_full, df_miss, attach, labels_ori
