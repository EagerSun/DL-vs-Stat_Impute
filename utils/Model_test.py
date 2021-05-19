import numpy as np
import pandas as pd

class Model_test():
    def __init__(self, label_reverse, label_ori, column_location, column_name, mode = 'one_hot'):
        self.mode = mode
        self.label_reverse = label_reverse
        self.label_ori = label_ori
        self.column_location = column_location
        self.column_name = column_name
        
       
    def return_accuary_for_con_cat(self, index, generate_i, mask_i, list_original_label):
        if self.label_reverse[index][0] == 'con':
            max_ = self.label_reverse[index][1][0]
            min_ = self.label_reverse[index][1][1]
            max_ori = self.label_ori[index][1][0]
            min_ori = self.label_ori[index][1][1]
            if self.mode == 'embedding':
                generate_i = (generate_i + 1.)/2.
            generate_i_re = generate_i * (max_ - min_) + min_
            generate_i_ori = (generate_i_re - min_ori)/(max_ori - min_ori)
            label_i_ori = np.array(list_original_label, np.float32).reshape(len(list_original_label), 1)
            if max_ori != min_ori:
                label_i_ori = (label_i_ori - min_ori)/(max_ori - min_ori)
            else:
                label_i_ori = (label_i_ori - min_ori) * 0.

            return ['con', np.sum(((generate_i_ori - label_i_ori)**2) * (1. - mask_i)), np.sum((1. - mask_i))]

        else:
            dictionary = self.label_reverse[index][1][1]
            generate_i_argmax = list(np.argmax(generate_i, axis = 1))
            mask_i_argmax = mask_i[:, 0]
            label_i_argmax = list_original_label
            result = np.array([dictionary[generate_i_argmax[i]] == label_i_argmax[i] for i in range(len(label_i_argmax))], dtype = np.float32)
            
            return ['cat', np.sum(result * (1. - mask_i_argmax)), np.sum((1. - mask_i_argmax))]
        
    def return_con_loss_cat_accuary_test_result(self, result):
        con_loss = 0
        con_mask_sum = 0
        cat_accuary = 0
        cat_mask_sum = 0
        
        for index, i in enumerate(result):
            if i[0] == 'con':
                con_loss = con_loss + i[1]
                con_mask_sum = con_mask_sum + i[2]
            else:
                cat_accuary = cat_accuary + i[1]
                cat_mask_sum = cat_mask_sum + i[2]
                
        if con_mask_sum == 0:
            con_mask_sum = con_mask_sum + 1  
        if cat_mask_sum == 0:
            cat_mask_sum = cat_mask_sum + 1
            
        return float(np.sqrt(con_loss/con_mask_sum)), float(cat_accuary/cat_mask_sum)
        
    def cross_validation_result(self, data_list, mask_list, df_full_list, df_miss_list):
        for index, i in enumerate(data_list):
            if index == 0:
                data = i
                mask = mask_list[index]
            else:
                data = np.concatenate((data, i), axis = 0)
                mask = np.concatenate((mask, mask_list[index]), axis = 0)
                
        df_full = pd.concat(df_full_list)
        df_miss = pd.concat(df_miss_list)
        return data, mask, df_full, df_miss
        
    def model_test(self, data = None, mask = None, df_original = None):
        result = []
        
        for index, i in enumerate(self.column_name):
            if index == 0:
                generate_i = data[:, 0:self.column_location[index]]
                mask_i = mask[:, 0:self.column_location[index]]
                list_original_label = df_original[i].to_list()
                re = self.return_accuary_for_con_cat(index = index, generate_i = generate_i, mask_i = mask_i, list_original_label = list_original_label)
                result.append(re)

            else:
                generate_i = data[:, self.column_location[index - 1]:self.column_location[index]]
                mask_i = mask[:, self.column_location[index - 1]:self.column_location[index]]
                list_original_label = df_original[i].to_list()
                re = self.return_accuary_for_con_cat(index = index, generate_i = generate_i, mask_i = mask_i, list_original_label = list_original_label)
                result.append(re)

        con_loss, cat_accuray = self.return_con_loss_cat_accuary_test_result(result = result)
        return con_loss, cat_accuray

    
