import os
import numpy as np
import pandas as pd
import gc
from data_process.return_data_miss_and_full import return_data_miss_and_full
from data_process.return_data_miss_and_full_R import return_data_miss_and_full_R
from utils.Model_test import Model_test
from utils.returnMedianrange import returnMedianrange

def return_mask_of_data(data):
    mask = 1. - np.isnan(data)   
    return mask
  
def Rimputed_test(miss_method, index_case, index_miss, imputed_method = 'MICE', num_of_test = 100):
    con_res = []
    cat_res = []
    for index in range(num_of_test):
        try:
            data, _, _, _, df_full, _, _, labels_ori = return_data_miss_and_full(miss_method = miss_method, index_case = index_case, index_miss = index_miss, index_file = index, mode = 'one_hot')
            
            mask = return_mask_of_data(data = data)
            #print("Now, Data Frames for Case {0} with full version, missing {1:.2%} are being prepared...".format(index_case, float(index_miss/100)))
            data_imputed, column_name, column_location, labels, _ = return_data_miss_and_full_R(miss_method = miss_method, index_case = index_case, index_miss = index_miss, index_file = index, imputed_method = imputed_method)

            mt = Model_test(label_reverse = labels, label_ori = labels_ori, column_location = column_location, column_name = column_name)
            con_loss, cat_acc = mt.model_test(data = data_imputed, mask = mask, df_original = df_full)
            con_res.append(con_loss)
            cat_res.append(cat_acc)
        except:
            continue
        
    #print("Now, the preprocessing of data had been finished.\n")
    returnMedianrange(con_loss_last_array = con_res, cat_accuracy_last_array = cat_res, num_of_test = num_of_test, miss_method = miss_method, index_case = index_case, index_miss = index_miss, name = imputed_method)
    gc.collect()
