import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.layers import Input
from utils.return_mask import return_mask_of_data
from utils.return_layer import return_layer
from utils.data_shuffle_noise import data_shuffle_noise
from utils.Model_test import Model_test
from utils.Performance_store import Performance_store
from parameters.Parameters_setting import parameters_setting
from tensorflow.python.keras.models import Sequential, Model
from data_process.return_data_miss_and_full import return_data_miss_and_full
from sklearn.model_selection import KFold

class GAIN(object):
    
    #The embedding size should be larger than or equal to 2. This is very important.
    #The num of fold should be 5, or 10. For 5, the fold_index should be picked from 0-4, for 10 should be 0-9.
    
    def __init__(self, miss_method, index_case, index_miss, batch_num, epoch, index_file, sampletest = False, index_pick = "continuous_first"):
        
        self.miss_method = miss_method
        self.index_case = index_case
        self.index_miss = index_miss
        self.index_file = index_file
        self.batch_size = batch_num
        self.index_pick = index_pick
        
        self.data, self.column_name, self.column_location, self.label_reverse, self.df_original, self.df_miss, self.attach, self.label_ori = return_data_miss_and_full(miss_method = self.miss_method, index_case = self.index_case, index_miss = self.index_miss, index_file = self.index_file, mode = 'one_hot', sampletest = sampletest)
        self.mask = return_mask_of_data(data = self.data)

        self.para = parameters_setting(model_name = 'GAIN', data_shape = self.data.shape[1], epoch = epoch)
        self.cross_validation, self.loss_mode, self.d_loss_mode, self.g_loss_mode, self.epoch, self.alpha, self.loss_balance, self.p_hint, self.noise_high_limit = self.para.return_parameters()
        self.dsn = data_shuffle_noise(mode = 'one_hot', noise_zero = False, high = self.noise_high_limit)
        self.model_estimate = Model_test(label_reverse = self.label_reverse, label_ori = self.label_ori, column_location = self.column_location, column_name = self.column_name, mode = 'one_hot')
        self.ps = Performance_store(miss_method = self.miss_method, index_case = self.index_case, index_miss = self.index_miss, index_file = self.index_file, label_reverse = self.label_reverse, label_ori = self.label_ori, column_location = self.column_location, column_name = self.column_name, name = 'GAIN', mode = 'one_hot', index_pick = self.index_pick)
        self.network_layer_G, self.network_layer_D = self.para.return_layer_size()
    
    def return_generative_network(self):
        input_ = Input(shape=(self.network_layer_G[0],))

        for i in range(len(self.network_layer_G) - 1):
                #[W_i, b_i] = self.return_weights_biases(input_size = self.network_layer_G[i], output_size = self.network_layer_G[i + 1])
                #list_parameters = list_parameters + [W_i, b_i]
            if i == 0:
                output = return_layer(layer_input = input_, output_size = self.network_layer_G[i + 1])
            elif i == len(self.network_layer_D) - 2:
                output = return_layer(layer_input = output, output_size = self.network_layer_G[i + 1], activation = 'sigmoid')
            else:
                output = return_layer(layer_input = output, output_size = self.network_layer_G[i + 1])

        model = Model(inputs = input_, outputs = output)

        return model
        
    def return_discriminator_network(self):
        input_ = Input(shape=(self.network_layer_D[0],))

        for i in range(len(self.network_layer_D) - 1):
                #[W_i, b_i] = self.return_weights_biases(input_size = self.network_layer_G[i], output_size = self.network_layer_G[i + 1])
                #list_parameters = list_parameters + [W_i, b_i]
            if i == 0:
                output = return_layer(layer_input = input_, output_size = self.network_layer_D[i + 1])
            elif i == len(self.network_layer_D) - 2:
                output = return_layer(layer_input = output, output_size = self.network_layer_D[i + 1], activation = 'sigmoid')
            else:
                output = return_layer(layer_input = output, output_size = self.network_layer_D[i + 1])

        model = Model(inputs = input_, outputs = output)

        return model

    def loss(self, gen_x, x, m):
        if self.loss_mode ==  'log_mse_masked':
            for index, i in enumerate(self.column_location):
                if self.label_reverse[index][0] == 'con':
                    if index == 0:
                        loss = tf.reduce_sum((gen_x[:, 0:i] * m[:, 0:i] - x[:, 0:i] * m[:, 0:i])**2)
                    else:
                        loss = loss + tf.reduce_sum((gen_x[:, self.column_location[index - 1]:i] * m[:, self.column_location[index - 1]:i] - x[:, self.column_location[index - 1]:i] * m[:, self.column_location[index - 1]:i])**2)
                else:
                    if index == 0:
                        loss_target = -x[:, 0:i] * m[:, 0:i] * tf.log(gen_x[:, 0:i] + 1e-8)
                        loss = self.loss_balance * tf.reduce_sum(loss_target)
                    else:
                        loss_target = -x[:, self.column_location[index - 1]:i] * m[:, self.column_location[index - 1]:i] * tf.log(gen_x[:, self.column_location[index - 1]:i] + 1e-8)
                        loss = loss + self.loss_balance * tf.reduce_sum(loss_target)

            loss = loss/(tf.reduce_sum(m) + 1e-8)
            
        else:
            loss = tf.reduce_sum((gen_x * m - x * m)**2)/(tf.reduce_sum(m) + 1e-8)
        return loss
    
    def d_loss(self, m, gen_m):
        if self.d_loss_mode == 'log_masked':
            loss = -tf.reduce_mean(m * tf.log(gen_m + 1e-8) + (1. - m) * tf.log(1. - gen_m + 1e-8))
        else:
            loss = tf.reduce_mean((m - gen_m)**2)
        return loss
        
    def g_loss(self, m, gen_m):
        if self.g_loss_mode == 'log_masked':
            loss = -tf.reduce_mean((1. - m) * tf.log(gen_m + 1e-8))
        elif self.g_loss_mode == 'log_complete_masked':
            loss = -tf.reduce_mean((1. - m) * tf.log(gen_m + 1e-8) + m * tf.log(gen_m + 1e-8))
        return loss
    
    def return_defined_network_for_mode(self):
        x, m, h = self.para.return_placeholder()
        genModel = self.return_generative_network()
        InputG = tf.concat(values = [x, m], axis = 1) 
        gen_xpre = genModel(InputG)
        gen_x = gen_xpre * (1. - m) + x * m
        disModel = self.return_discriminator_network()
        InputD = tf.concat(values = [gen_x, h], axis = 1) 
        gen_m = disModel(InputD)
        D_loss = self.d_loss(m = m, gen_m = gen_m)
        G_loss_p1 = self.g_loss(m = m, gen_m = gen_m)
        G_loss_p2 = self.loss(gen_x = gen_xpre, x = x, m = m)
        G_loss = (G_loss_p1 + self.alpha * G_loss_p2)
        G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = genModel.trainable_weights)
        D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = disModel.trainable_weights)

        return G_loss, D_loss, G_loss_p2, G_solver, D_solver, gen_xpre, x, m, h
            
    def return_hint_of_mask(self, train_m_batch):
        mask_hint = np.random.uniform(size = (train_m_batch.shape[0], self.mask.shape[1]), low = 0.0, high = 1.0)
        mask_res = np.float32(mask_hint > 1 - self.p_hint)
        return mask_res * train_m_batch
    
    def return_Kfold(self, tr, te):
        data = self.data.copy()
        mask = self.mask.copy()
        df_full = self.df_original.copy()
        df_miss = self.df_miss.copy()
            
        train_d = data[tr]
        test_d = data[te]
        train_m = mask[tr]
        test_m = mask[te]
        df_train_label = df_full.iloc[tr].copy().reset_index(drop = True)
        df_test_label = df_full.iloc[te].copy().reset_index(drop = True)
        df_train_miss = df_miss.iloc[tr].copy().reset_index(drop = True)
        df_test_miss = df_miss.iloc[te].copy().reset_index(drop = True)
        return train_d, train_m, df_train_label, df_train_miss, test_d, test_m, df_test_label, df_test_miss
    
    def train_process(self):
        if self.cross_validation is None:
            G_loss, D_loss, G_loss_p2, G_solver, D_solver, gen_x, x, m, h = self.return_defined_network_for_mode()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())

            for epoc in range(self.epoch):

                train_d, train_m = self.dsn.data_shuffle(train_data = self.data.copy(), train_mask = self.mask.copy())
                for iteration in range(int(train_d.shape[0]/self.batch_size)):

                    train_d_batch = train_d[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    train_m_batch = train_m[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    train_h_batch = self.return_hint_of_mask(train_m_batch = train_m_batch)

                    _ = sess.run([G_solver], feed_dict = {x: train_d_batch, m: train_m_batch, h: train_h_batch})
                    _ = sess.run([D_solver], feed_dict = {x: train_d_batch, m: train_m_batch, h: train_h_batch})
            
            data_noised = self.dsn._add_noise_(train_data = self.data.copy(), train_mask = self.mask.copy())
            data_imputed = sess.run(gen_x, feed_dict = {x: data_noised, m: self.mask})
            data_imputed = data_imputed * (1. - self.mask) + data_noised * self.mask
            con_loss, cat_accuracy = self.model_estimate.model_test(data = data_imputed, mask = self.mask.copy(), df_original = self.df_original)
            self.ps.storeImputed(data = data_imputed, df_original = self.df_original, df_miss = self.df_miss)
            sess.close()
            tf.keras.backend.clear_session()
            
            return con_loss, cat_accuracy
        else: 
            data_list = []
            mask_list = []
            df_full_list = []
            df_miss_list = []
            
            kf = KFold(n_splits=self.cross_validation, shuffle = True)
            kf_index = np.arange(0, self.data.shape[0])
            for tr, te in kf.split(kf_index):
                train_d, train_m, df_train_label, df_train_miss, test_d, test_m, df_test_label, df_test_miss = self.return_Kfold(tr, te)
                G_loss, D_loss, G_loss_p2, G_solver, D_solver, gen_x, x, m, h = self.return_defined_network_for_mode()
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
                for epoc in range(self.epoch):

                    train_d_, train_m_ = self.dsn.data_shuffle(train_data = train_d.copy(), train_mask = train_m.copy())
                    for iteration in range(int(train_d.shape[0]/self.batch_size)):

                        train_d_batch = train_d_[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                        train_m_batch = train_m_[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                        train_h_batch = self.return_hint_of_mask(train_m_batch = train_m_batch)

                        _ = sess.run([G_solver], feed_dict = {x: train_d_batch, m: train_m_batch, h: train_h_batch})
                        _ = sess.run([D_solver], feed_dict = {x: train_d_batch, m: train_m_batch, h: train_h_batch})
                
                test_d = self.dsn._add_noise_(train_data = test_d, train_mask = test_m)
                test_d_imputed = sess.run(gen_x, feed_dict = {x: test_d, m: test_m})
                test_d = test_d_imputed * (1. - test_m) + test_d * test_m
                data_list.append(test_d)
                mask_list.append(test_m)
                df_full_list.append(df_test_label)
                df_miss_list.append(df_test_miss)
                                   
                sess.close()
                #tf.reset_default_graph()
                tf.keras.backend.clear_session()
                
            data_c, mask_c, df_original_c, df_miss_c = self.model_estimate.cross_validation_result(data_list, mask_list, df_full_list, df_miss_list)
            con_loss, cat_accuracy = self.model_estimate.model_test(data = data_c, mask = mask_c, df_original = df_original_c)
            self.ps.storeImputed(data = data_c, df_original = df_original_c, df_miss = df_miss_c)

            ###
            return con_loss, cat_accuracy
    
    def train_process_sample(self):
        if self.cross_validation is None:
            G_loss, D_loss, G_loss_p2, G_solver, D_solver, gen_x, x, m, h = self.return_defined_network_for_mode()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            
            con_list, cat_list = [], []
            for epoc in range(self.epoch):

                train_d, train_m = self.dsn.data_shuffle(train_data = self.data.copy(), train_mask = self.mask.copy())
                for iteration in range(int(train_d.shape[0]/self.batch_size)):

                    train_d_batch = train_d[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    train_m_batch = train_m[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    train_h_batch = self.return_hint_of_mask(train_m_batch = train_m_batch)

                    _ = sess.run([G_solver], feed_dict = {x: train_d_batch, m: train_m_batch, h: train_h_batch})
                    _ = sess.run([D_solver], feed_dict = {x: train_d_batch, m: train_m_batch, h: train_h_batch})
            
                data_noised = self.dsn._add_noise_(train_data = self.data.copy(), train_mask = self.mask.copy())
                data_imputed = sess.run(gen_x, feed_dict = {x: data_noised, m: self.mask})
                data_imputed = data_imputed * (1. - self.mask) + data_noised * self.mask
                con_loss, cat_accuracy = self.model_estimate.model_test(data = data_imputed, mask = self.mask.copy(), df_original = self.df_original)
                con_list.append(con_loss)
                cat_list.append(cat_accuracy)
                
            index_re = self.ps.return_con_cat_index(con_list = con_list, cat_accuracy = cat_list)
            sess.close()
            tf.keras.backend.clear_session()
            
            return index_re
        else: 
            index_re_list = []
            
            kf = KFold(n_splits=self.cross_validation, shuffle = True)
            kf_index = np.arange(0, self.data.shape[0])
            for tr, te in kf.split(kf_index):
                train_d, train_m, df_train_label, df_train_miss, test_d, test_m, df_test_label, df_test_miss = self.return_Kfold(tr, te)
                G_loss, D_loss, G_loss_p2, G_solver, D_solver, gen_x, x, m, h = self.return_defined_network_for_mode()
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
                
                con_list, cat_list = [], []
                
                for epoc in range(self.epoch):

                    train_d_, train_m_ = self.dsn.data_shuffle(train_data = train_d.copy(), train_mask = train_m.copy())
                    for iteration in range(int(train_d.shape[0]/self.batch_size)):

                        train_d_batch = train_d_[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                        train_m_batch = train_m_[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                        train_h_batch = self.return_hint_of_mask(train_m_batch = train_m_batch)

                        _ = sess.run([G_solver], feed_dict = {x: train_d_batch, m: train_m_batch, h: train_h_batch})
                        _ = sess.run([D_solver], feed_dict = {x: train_d_batch, m: train_m_batch, h: train_h_batch})
                
                    test_d = self.dsn._add_noise_(train_data = test_d, train_mask = test_m)
                    test_d_imputed = sess.run(gen_x, feed_dict = {x: test_d, m: test_m})
                    test_d = test_d_imputed * (1. - test_m) + test_d * test_m
                    con_loss, cat_accuracy = self.model_estimate.model_test(data = test_d, mask = test_m, df_original = df_test_label)
                    con_list.append(con_loss)
                    cat_list.append(cat_accuracy)
                    
                index_re = self.ps.return_con_cat_index(con_list = con_list, cat_accuracy = cat_list)  
                index_re_list.append(index_re)
                sess.close()
                #tf.reset_default_graph()
                tf.keras.backend.clear_session()

            ###
            return np.around(np.mean(index_re_list))
        