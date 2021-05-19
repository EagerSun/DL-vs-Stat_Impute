import numpy as np
import pandas as pd
import os

class Performance_store():
    def __init__(self, miss_method, index_case, index_miss, index_file, label_reverse, label_ori, column_location, column_name, name, mode = 'one_hot', index_pick = 'continuous_first'):
        self.mode = mode
        self.miss_method = miss_method
        self.index_case = index_case
        self.index_miss = index_miss
        self.index_file = index_file
        self.label_reverse = label_reverse
        self.label_ori = label_ori
        self.column_location = column_location
        self.column_name = column_name
        self.name = name
        self.index_pick = index_pick
        
       
    def returnDfColumn(self, index, generate_i):
        if self.label_reverse[index][0] == 'con':
            max_ = self.label_reverse[index][1][0]
            min_ = self.label_reverse[index][1][1]
            if self.mode == 'embedding':
                generate_i = (generate_i + 1.)/2.
            generate_i_ori = generate_i * (max_ - min_) + min_
            generate_i_ori = generate_i_ori.reshape(generate_i_ori.shape[0], )
            return list(generate_i_ori)
        
        else:
            dictionary = self.label_reverse[index][1][1]
            generate_i_argmax = list(np.argmax(generate_i, axis = 1))
            generate_i_ori = [dictionary[generate_i_argmax[i]] for i in range(len(generate_i_argmax))]
                
            return generate_i_ori
        
    def returnImputed(self, data = None):
        
        df_res = pd.DataFrame()
        for index, i in enumerate(self.column_name):
            if index == 0:
                generate_i = data[:, 0:self.column_location[index]]
                col = self.returnDfColumn(index = index, generate_i = generate_i)
                df_res[i] = col
                
            else:
                generate_i = data[:, self.column_location[index - 1]:self.column_location[index]]
                col = self.returnDfColumn(index = index, generate_i = generate_i)
                df_res[i] = col
                
        return df_res
    
    def storeImputed(self, data = None, df_original = None, df_miss = None):
        
        main_path = os.path.join(os.getcwd(), 'data_stored', 'data_imputed')
        if not os.path.exists(main_path):
            os.mkdir(main_path)
        method_path = os.path.join(main_path, self.miss_method)
        if not os.path.exists(method_path):
            os.mkdir(method_path)
        case_path = os.path.join(method_path, 'Case{0}'.format(self.index_case))
        if not os.path.exists(case_path):
            os.mkdir(case_path)
        miss_path = os.path.join(case_path, 'miss{0}'.format(self.index_miss))
        if not os.path.exists(miss_path):
            os.mkdir(miss_path)
        model_path = os.path.join(miss_path, self.name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        original_path = os.path.join(model_path, 'original')
        if not os.path.exists(original_path):
            os.mkdir(original_path)
        missd_path = os.path.join(model_path, 'miss')
        if not os.path.exists(missd_path):
            os.mkdir(missd_path)
        imputed_path = os.path.join(model_path, 'imputed')
        if not os.path.exists(imputed_path):
            os.mkdir(imputed_path)
        
        df_original.to_csv(os.path.join(original_path, '{0}.csv'.format(self.index_file)), index = False)
        df_miss.to_csv(os.path.join(missd_path, '{0}.csv'.format(self.index_file)), index = False)
        df_imputed = self.returnImputed(data = data)
        df_imputed.to_csv(os.path.join(imputed_path, '{0}.csv'.format(self.index_file)), index = False)
        
    def return_con_cat_index(self, con_list, cat_accuracy):
        if self.index_pick == 'continuous_first':
            index = con_list.index(min(con_list)) + 1
        elif self.index_pick == 'categorical_first':
            index = cat_accuracy.index(max(cat_accuracy)) + 1
            
        return index
        

    
