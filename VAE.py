import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.layers import Input, Concatenate
from tensorflow.python.keras.models import Sequential, Model
import tensorflow_probability as tfp
from utils.return_mask import return_mask_of_data
from utils.return_layer import return_layer
from utils.data_shuffle_noise import data_shuffle_noise
from utils.Model_test import Model_test
from utils.Performance_store import Performance_store
from parameters.Parameters_setting import parameters_setting
from data_process.return_data_miss_and_full import return_data_miss_and_full
from sklearn.model_selection import KFold

class VAE(object):
    
    #The embedding size should be larger than or equal to 2. This is very important.
    #The num of fold should be 5, or 10. For 5, the fold_index should be picked from 0-4, for 10 should be 0-9.
    
    def __init__(self, miss_method, index_case, index_miss, index_file, epoch, batch_num, sampletest = False, index_pick = "continuous_first"):
        #Here, we define the data species, data encoding method, and whether to use cross_validation here...
        self.miss_method = miss_method
        self.index_case = index_case
        self.index_miss = index_miss
        self.index_file = index_file
        self.batch_size = int(batch_num)
        self.index_pick = index_pick
        
        #Return the train data and mask as we need.
        self.data, self.column_name, self.column_location, self.label_reverse, self.df_original, self.df_miss, self.attach, self.label_ori = return_data_miss_and_full(miss_method = self.miss_method, index_case = self.index_case, index_miss = self.index_miss, index_file = self.index_file, mode = 'one_hot', sampletest = sampletest)
        self.mask = return_mask_of_data(data = self.data)
  
        self.para = parameters_setting(model_name = 'VAE', data_shape = self.data.shape[1], epoch = epoch)
        self.cross_validation, self.loss_mode, self.kl_loss_mode, self.loss_axis, self.train_with_complete, self.noise_zero, self.test_iteration, self.epoch, self.alpha, self.loss_balance, self.noise_high_limit, self.learning_rate = self.para.return_parameters()
        self.dsn = data_shuffle_noise(mode = 'one_hot', noise_zero = self.noise_zero, high = self.noise_high_limit)
        self.model_estimate = Model_test(label_reverse = self.label_reverse, label_ori = self.label_ori, column_location = self.column_location, column_name = self.column_name, mode = 'one_hot')
        self.ps = Performance_store(miss_method = self.miss_method, index_case = self.index_case, index_miss = self.index_miss, index_file = self.index_file, label_reverse = self.label_reverse, label_ori = self.label_ori, column_location = self.column_location, column_name = self.column_name, name = 'VAE', mode = 'one_hot', index_pick = self.index_pick)
        self.latent_size, self.network_layer_G, self.network_latent_layer_G, self.network_layer_D, self.network_latent_layer_D = self.para.return_layer_size()
    
    def return_encode_network_divide(self, x, output_mode = 'mean'):
        
        if output_mode == 'mean':
            inputs = x
            if len(self.network_latent_layer_G) == 1:
                output = return_layer(layer_input = inputs, output_size = self.network_latent_layer_G[0], activation = 'tanh')
            else:
                for i in range(len(self.network_latent_layer_G)):
                    if i == 0:
                        output = return_layer(layer_input = inputs, output_size = self.network_latent_layer_G[i], activation = 'tanh')
                    elif i == len(self.network_latent_layer_G) - 1:
                        output = return_layer(layer_input = output, output_size = self.network_latent_layer_G[i], activation = 'tanh')
                    else:
                        output = return_layer(layer_input = output, output_size = self.network_latent_layer_G[i], activation = 'tanh')

                
        else:
            inputs = x
            if len(self.network_latent_layer_G) == 1:
                output = return_layer(layer_input = inputs, output_size = self.network_latent_layer_G[0], activation = 'tanh')
            else:
                for i in range(len(self.network_latent_layer_G)):
                    if i == 0:
                        output = return_layer(layer_input = inputs, output_size = self.network_latent_layer_G[i], activation = 'tanh')
                    elif i == len(self.network_latent_layer_G) - 1:
                        output = return_layer(layer_input = output, output_size = self.network_latent_layer_G[i], activation = 'tanh')
                    else:
                        output = return_layer(layer_input = output, output_size = self.network_latent_layer_G[i], activation = 'tanh') 
        return output

    def return_encode_network(self):
        
        inputs = Input(shape=(self.network_layer_G[0],))

        for i in range(len(self.network_layer_G)):    
            if i == 0:
                output = return_layer(layer_input = inputs, output_size = self.network_layer_G[i + 1], activation = 'tanh')

            elif i == len(self.network_layer_G) - 1:
                output_p1 = self.return_encode_network_divide(x = output, output_mode = 'mean')
                output_p2 = self.return_encode_network_divide(x = output, output_mode = 'var')

                output = Concatenate(axis=1)([output_p1, output_p2])

            else:
                output = return_layer(layer_input = output, output_size = self.network_layer_G[i + 1], activation = 'tanh')

        model = Model(inputs = inputs, outputs = output)

        return model

               
    def return_decode_network(self):

        inputs = Input(shape=(self.network_layer_D[0],))
        for i in range(len(self.network_layer_D) - 1):

            if i == 0:
                output = return_layer(layer_input = inputs, output_size = self.network_layer_D[i + 1], activation = 'tanh')
            elif i == len(self.network_layer_D) - 2:
                output = return_layer(layer_input = output, output_size = self.network_layer_D[i + 1], activation = 'tanh')
            else:
                output = return_layer(layer_input = output, output_size = self.network_layer_D[i + 1], activation = 'tanh')
                
        model = Model(inputs = inputs, outputs = output)
        
        return model
    
    def return_decode_network_divide(self, output_mode = 'mean'):
        
        if output_mode == 'mean':
            inputs = Input(shape=(self.network_layer_D[-1],))
            if len(self.network_latent_layer_D) == 1:
                output = return_layer(layer_input = inputs, output_size = self.network_latent_layer_D[0], activation = 'sigmoid')
            else:
                for i in range(len(self.network_latent_layer_D)):
                    if i == 0:
                        output = return_layer(layer_input = inputs, output_size = self.network_latent_layer_D[i], activation = 'relu')
                    elif i == len(self.network_latent_layer_D) - 1:
                        output = return_layer(layer_input = output, output_size = self.network_latent_layer_D[i], activation = 'sigmoid')
                    else:
                        output = return_layer(layer_input = output, output_size = self.network_latent_layer_D[i], activation = 'relu')
     
        else:
            inputs = Input(shape=(self.network_layer_D[-1],))
            if len(self.network_latent_layer_D) == 1:
                output = return_layer(layer_input = inputs, output_size = self.network_latent_layer_D[0], activation = 'tanh')
            else:
                for i in range(len(self.network_latent_layer_D)):
                    if i == 0:
                        output = return_layer(layer_input = inputs, output_size = self.network_latent_layer_D[i], activation = 'tanh')
                    elif i == len(self.network_latent_layer_D) - 1:
                        output = return_layer(layer_input = output, output_size = self.network_latent_layer_D[i], activation = 'tanh')
                    else:
                        output = return_layer(layer_input = output, output_size = self.network_latent_layer_D[i], activation = 'tanh')
        model = Model(inputs = inputs, outputs = output)
        return model
        
    def loss(self, distribution, x, m):
        if self.loss_mode == 'log_masked':
            prob = distribution.log_prob(x)
            loss = -tf.reduce_sum(prob * m, axis = self.loss_axis)/tf.reduce_sum(m, axis = self.loss_axis)
        elif self.loss_mode == 'log':
            prob = distribution.log_prob(x)
            loss = -tf.reduce_mean(prob, axis = self.loss_axis)
            
        return loss
                 
    def kl_loss(self, output_from_generative, p_z, q_z):
        if self.kl_loss_mode == 'sample':
            loss = -1/2. * tf.reduce_mean(1. + output_from_generative[:, self.latent_size:] - output_from_generative[:, :self.latent_size]**2 + tf.math.exp(output_from_generative[:, self.latent_size:]))
        else:
            
            loss = tf.reduce_mean(tfp.distributions.kl_divergence(p_z, q_z), axis = self.loss_axis)
            
        return loss
        
        
    def return_cos_similarity(self, x, label, mode = 'tanh'):
        normalized_x, _ = tf.linalg.normalize(x, axis = 1)
        normalized_label, _ = tf.linalg.normalize(label[0][1:], axis = 1)
        cos_similarity_matrix = tf.matmul(normalized_x, tf.transpose(normalized_label))
        if mode == 'non':
            cos_similarity_matrix = cos_similarity_matrix
        elif mode == 'relu':
            cos_similarity_matrix = tf.nn.relu(cos_similarity_matrix)
        elif mode == 'softmax':
            cos_similarity_matrix = tf.nn.softmax(cos_similarity_matrix)
        elif mode == 'sigmoid':
            cos_similarity_matrix = tf.nn.sigmoid(cos_similarity_matrix)
        elif mode == 'tanh':
            cos_similarity_matrix = tf.math.tanh(cos_similarity_matrix)
        
        return cos_similarity_matrix
    
    
    def return_defined_network_for_mode(self):
        x, m = self.para.return_placeholder()
        model_e = self.return_encode_network()
        gen_p = model_e(x)
        distribution_p = tfp.distributions.Normal(loc=gen_p[:, :self.latent_size], scale=tf.exp(gen_p[:, self.latent_size:]))
        model_d = self.return_decode_network()
        gen_x = model_d(distribution_p.sample())
        distribution_q = tfp.distributions.Normal(loc=np.zeros(self.latent_size, dtype=np.float32), scale=np.ones(self.latent_size, dtype=np.float32))
        model_ddm = self.return_decode_network_divide(output_mode = 'mean')
        model_ddv = self.return_decode_network_divide(output_mode = 'var')
        gen_x_mean = model_ddm(gen_x)
        gen_x_var = model_ddv(gen_x)
        distribution = tfp.distributions.Normal(loc=gen_x_mean, scale=tf.exp(gen_x_var))
        loss_p2 = self.kl_loss(output_from_generative = gen_p, p_z = distribution_p, q_z = distribution_q)
        loss_p1 = self.loss(distribution = distribution, x = x, m = m)
        loss = tf.reduce_sum(loss_p1 + self.alpha * loss_p2)
        #loss = tf.reduce_sum(self.alpha * loss_p1)
        solver = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(loss, var_list = model_e.trainable_variables + model_d.trainable_variables + model_ddm.trainable_variables + model_ddv.trainable_variables)


        return loss, loss_p1, loss_p2, solver, gen_x_mean, x, m
           
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
            loss, loss_p1, loss_p2, solver, gen_x, x, m = self.return_defined_network_for_mode()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            if self.train_with_complete:
                train_data, train_mask = self.dsn.dataNonenan(data = self.data.copy(), mask = self.mask.copy())
            else:
                train_data, train_mask = self.data.copy(), self.mask.copy()

            for epoc in range(self.epoch):

                train_d, train_m = self.dsn.data_shuffle(train_data, train_mask)

                for iteration in range(int(train_d.shape[0]/self.batch_size)):

                    train_d_batch = train_d[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    train_m_batch = train_m[iteration * self.batch_size : (iteration + 1) * self.batch_size]

                    _ = sess.run([solver], feed_dict = {x: train_d_batch, m: train_m_batch})

            data_noised = self.dsn._add_noise_(train_data = self.data.copy(), train_mask = self.mask.copy())
            data_imputed = self.testIteration(sess = sess, gen_x = gen_x, data = data_noised, mask = self.mask.copy(), x = x, m = m)
            data_generate = data_imputed * (1. - self.mask) + data_noised * self.mask
            con_loss, cat_accuracy = self.model_estimate.model_test(data = data_generate, mask = self.mask.copy(), df_original = self.df_original)
            self.ps.storeImputed(data = data_generate, df_original = self.df_original, df_miss = self.df_miss)
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
                loss, loss_p1, loss_p2, solver, gen_x, x, m = self.return_defined_network_for_mode()
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
                if self.train_with_complete:
                    train_d, train_m = self.dataNonenan(data = train_d.copy(), mask = train_m.copy())
                else:
                    train_d, train_m = train_d, train_m 
            
                for epoc in range(self.epoch):

                    train_d_, train_m_ = self.dsn.data_shuffle(train_data = train_d, train_mask = train_m)
                    #print(train_d.shape[0])
                    for iteration in range(int(train_d.shape[0]/self.batch_size)):

                        train_d_batch = train_d_[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                        train_m_batch = train_m_[iteration * self.batch_size : (iteration + 1) * self.batch_size]

                        _ = sess.run([solver], feed_dict = {x: train_d_batch, m: train_m_batch})
                test_d = self.dsn._add_noise_(train_data = test_d, train_mask = test_m)
                test_d_imputed = self.testIteration(sess = sess, gen_x = gen_x, data = test_d.copy(), mask = test_m, x = x, m = m)
                test_d = test_d_imputed * (1. - test_m) + test_d * test_m
                data_list.append(test_d)
                mask_list.append(test_m)
                df_full_list.append(df_test_label)
                df_miss_list.append(df_test_miss)
                sess.close()
                #tf.reset_default_graph()
                tf.keras.backend.clear_session()

            data_c, mask_c, df_original_c, df_miss_c = self.model_estimate.cross_validation_result(data_list, mask_list, df_full_list, df_miss_list)
            con_loss, cat_accuracy = self.model_estimate.model_test(data_c, mask_c, df_original_c)
            self.ps.storeImputed(data = data_c, df_original = df_original_c, df_miss = df_miss_c)
            return con_loss, cat_accuracy
        
    def train_process_sample(self):
        if self.cross_validation is None:
            loss, loss_p1, loss_p2, solver, gen_x, x, m = self.return_defined_network_for_mode()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            
            con_list, cat_list = [], []
            
            if self.train_with_complete:
                train_data, train_mask = self.dsn.dataNonenan(data = self.data.copy(), mask = self.mask.copy())
            else:
                train_data, train_mask = self.data.copy(), self.mask.copy()

            for epoc in range(self.epoch):

                train_d, train_m = self.dsn.data_shuffle(train_data, train_mask)

                for iteration in range(int(train_d.shape[0]/self.batch_size)):

                    train_d_batch = train_d[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    train_m_batch = train_m[iteration * self.batch_size : (iteration + 1) * self.batch_size]

                    _ = sess.run([solver], feed_dict = {x: train_d_batch, m: train_m_batch})

                data_noised = self.dsn._add_noise_(train_data = self.data.copy(), train_mask = self.mask.copy())
                data_imputed = self.testIteration(sess = sess, gen_x = gen_x, data = data_noised, mask = self.mask.copy(), x = x, m = m)
                data_generate = data_imputed * (1. - self.mask) + data_noised * self.mask
                con_loss, cat_accuracy = self.model_estimate.model_test(data = data_generate, mask = self.mask.copy(), df_original = self.df_original)
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
                loss, loss_p1, loss_p2, solver, gen_x, x, m = self.return_defined_network_for_mode()
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
                
                con_list, cat_list = [], []
                
                if self.train_with_complete:
                    train_d, train_m = self.dataNonenan(data = train_d.copy(), mask = train_m.copy())
                else:
                    train_d, train_m = train_d, train_m 
            
                for epoc in range(self.epoch):

                    train_d_, train_m_ = self.dsn.data_shuffle(train_data = train_d, train_mask = train_m)
                    #print(train_d.shape[0])
                    for iteration in range(int(train_d.shape[0]/self.batch_size)):

                        train_d_batch = train_d_[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                        train_m_batch = train_m_[iteration * self.batch_size : (iteration + 1) * self.batch_size]

                        _ = sess.run([solver], feed_dict = {x: train_d_batch, m: train_m_batch})
                        
                    test_d = self.dsn._add_noise_(train_data = test_d, train_mask = test_m)
                    test_d_imputed = self.testIteration(sess = sess, gen_x = gen_x, data = test_d.copy(), mask = test_m, x = x, m = m)
                    test_d = test_d_imputed * (1. - test_m) + test_d * test_m
                    con_loss, cat_accuracy = self.model_estimate.model_test(data = test_d, mask = test_m, df_original = df_test_label)
                    con_list.append(con_loss)
                    cat_list.append(cat_accuracy)
                    
                index_re = self.ps.return_con_cat_index(con_list = con_list, cat_accuracy = cat_list)  
                index_re_list.append(index_re)
                sess.close()
                #tf.reset_default_graph()
                tf.keras.backend.clear_session()

            return np.around(np.mean(index_re_list))
        
    def testIteration(self, sess, gen_x, data = None, mask = None, x = None, m = None):
        if self.test_iteration is None:
            generate_data = sess.run(gen_x, feed_dict = {x: data, m: mask})
        else:
            generate_data = data
            for i in range(self.test_iteration):
                generate_data = sess.run(gen_x, feed_dict = {x: generate_data, m: mask})
                
        return generate_data
