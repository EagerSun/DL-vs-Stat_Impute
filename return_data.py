import os
import numpy as np
import pandas as pd


def return_data_file(index_case):
    data_path = os.path.join(os.getcwd(), 'data_stored', 'data')
    data_path_detail = [os.path.join(data_path, i) for i in os.listdir(data_path) if i != '.DS_Store']
    data_names_detail = [j for j in os.listdir(data_path) if j != '.DS_Store']
    for index, i in enumerate(data_names_detail):
        name = i.split('.')[0]
        if str(index_case) == name:
            location_index = index
            break
        else:
            continue
            
    data_address_case = data_path_detail[location_index]
    data_names_case = data_names_detail[location_index]
    
    df_full = pd.read_csv(data_address_case)
            
    return df_full

def return_split_con_cat_column(df_full):
    columns = df_full.columns
    columns_split = [i.split('_') for i in columns]
    con = []
    cat = []
    for index, i in enumerate(columns_split):
        if i[0][0:3] == 'con':
            con.append(columns[index])
        elif i[0][0:3] == 'cat':
            cat.append(columns[index])
        else:
            whatever = 0
    
    return con, cat

def return_dictionary_column(column_unique):
    dict_res = {}
    dict_rev = {}
    for index, i in enumerate(column_unique):
        dict_res[i] = index
        dict_rev[index] = i
    
    return dict_res, dict_rev 

def return_encode_column(column_i, dictionary, mode):
    column_encode = []
    for index, i in enumerate(column_i):
        column_encode.append([dictionary[i]])

    array_encode = np.array(column_encode, dtype = np.float32)
        
    return array_encode
    
    
def return_normalized_column(column_i, mode):
    max_i_ori = np.max(np.array(column_i))
    min_i_ori = np.min(np.array(column_i))
    if max_i_ori != min_i_ori:
        column_m_normalize = [(i - min_i_ori)/(max_i_ori - min_i_ori) for i in column_i]
        column_m_normalize = [i for i in column_m_normalize]
    else:
        column_m_normalize = [i * 0. for i in column_i]
   
    array_m_normalized = np.array(column_m_normalize, np.float32).reshape(len(column_m_normalize), 1)
    
    return array_m_normalized, max_i_ori, min_i_ori

def return_data(index_case, mode = 'one_hot'):
    #print("Now, Data Frames for Case {0} with full version, missing {1:.2%} are being prepared...".format(index_case, float(index_miss/100)))
    df_full = return_data_file(index_case)
    con, cat = return_split_con_cat_column(df_full = df_full)
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
            column_i = df_full[i].to_list()
            columns_i_unique = list(set([i for i in column_i if str(i) != 'nan']))
            dict_res_column, dict_rev_column = return_dictionary_column(column_unique = columns_i_unique)
            column_encode = return_encode_column(column_i = column_i, dictionary = dict_res_column, mode = mode)
            locations.append(column_encode.shape[1])
            labels.append(['cat', [dict_res_column, dict_rev_column]])
            labels_ori.append(['cat', [dict_res_column, dict_rev_column]])
            attach.append(['cat', list(set([i for i in df_full[i].to_list() if str(i) != 'nan']))]) 
        elif i in con:
            column_i = df_full[i].to_list()
            column_encode, max_i_ori, min_i_ori = return_normalized_column(column_i = column_i, mode = mode)
            locations.append(column_encode.shape[1])
            labels.append(['con', [max_i_ori, min_i_ori]])
            labels_ori.append(['con', [max_i_ori, min_i_ori]])
            attach.append(['con', [max_i_ori, min_i_ori]])
            
        array_encode_column = np.array(column_encode, np.float32)

        if index == 0:
            array_result = array_encode_column
        else:
            array_result = np.concatenate((array_result, array_encode_column), axis = 1)
            locations[-1] = locations[-1] + locations[-2]
    #print("Now, the preprocessing of data had been finished.\n")   
    return array_result, df_full
