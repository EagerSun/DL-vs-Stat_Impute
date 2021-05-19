import os
import sys
import numpy as np
import pandas as pd

def returnMedianrange(con_loss_last_array, cat_accuracy_last_array, num_of_test, miss_method, index_case, index_miss, name):
    
    con_loss_last_array = [i for i in con_loss_last_array if str(i) != 'nan']
    cat_accuracy_last_array = [i for i in cat_accuracy_last_array if str(i) != 'nan']
    
    main_path = os.path.join(os.getcwd(), 'performance')
    if not os.path.exists(main_path):
        os.mkdir(main_path)
    method_path = os.path.join(main_path, miss_method)
    if not os.path.exists(method_path):
        os.mkdir(method_path)
    case_path = os.path.join(method_path, 'Case{0}'.format(index_case))
    if not os.path.exists(case_path):
        os.mkdir(case_path)
    miss_path = os.path.join(case_path, 'miss{0}'.format(index_miss))
    if not os.path.exists(miss_path):
        os.mkdir(miss_path)
    perform_path = os.path.join(miss_path, name + '_performance.csv')
    perform_summary_path = os.path.join(miss_path, name + '_summary.csv')
    
    #df_res = pd.DataFrame()
    #df_res['con_loss'] = con_loss_last_array
    #df_res['cat_accuracy'] = cat_accuracy_last_array
    #df_res.to_csv(perform_path, index = False)

    con_loss_last_array.sort()
    cat_accuracy_last_array.sort()
    num_of_test = len(con_loss_last_array)
    medCon = np.median(con_loss_last_array)
    medCat = np.median(cat_accuracy_last_array)
    q1Con = np.median(con_loss_last_array[:int(num_of_test/2)])
    q2Con = np.median(con_loss_last_array[int(num_of_test/2):])
    q1Cat = np.median(cat_accuracy_last_array[:int(num_of_test/2)])
    q2Cat = np.median(cat_accuracy_last_array[int(num_of_test/2):])

    meanCon = np.mean(con_loss_last_array)
    varCon = np.std(con_loss_last_array)
    meanCat = np.mean(cat_accuracy_last_array)
    varCat = np.std(cat_accuracy_last_array)

    df_rem = pd.DataFrame()
    df_rem['medCon'] = [medCon]
    df_rem['qRCon'] = [q2Con - q1Con]
    df_rem['medCat'] = [medCat]
    df_rem['qRCat'] = [q2Cat - q1Cat]

    df_rem['meanCon'] = [meanCon]
    df_rem['varCon'] = [varCon]
    df_rem['meanCat'] = [meanCat]
    df_rem['varCat'] = [varCat]
    df_rem.to_csv(perform_summary_path, index = False)
