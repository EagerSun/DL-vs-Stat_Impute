import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.layers import Input, Concatenate
from tensorflow.python.keras.models import Sequential, Model, load_model
import tensorflow_probability as tfp
from utils.return_mask import return_mask_of_data, return_array_size_and_location_for_columns_embedding
from utils.return_layer import return_layer, return_embedding_model_for_columns
from utils.data_shuffle_noise import data_shuffle_noise
from utils.Model_test import Model_test
from utils.Performance_store import Performance_store
from parameters.Parameters_setting import parameters_setting
from data_process.return_data_miss_and_full import return_data_miss_and_full
from sklearn.model_selection import KFold

class VAE_embedding(object):
    
    #The embedding size should be larger than or equal to 2. This is very important.
    #The num of fold should be 5, or 10. For 5, the fold_index should be picked from 0-4, for 10 should be 0-9.
    
    def __init__(self, miss_method, index_case, index_miss, index_file, epoch, batch_num, sampletest = False, index_pick = "continuous_first"):
        #Here, we define the data species, data encoding method, and whether to use cross_validation here...
        self.miss_method = miss_method
        self.index_case = index_case
        self.index_miss = index_miss
        self.batch_size = int(batch_num)
        self.index_file = index_file
        self.index_pick = index_pick
        
        #Return the train data and mask as we need.
        self.data, self.column_name, self.column_location, self.label_reverse, self.df_original, self.df_miss, self.attach, self.label_ori = return_data_miss_and_full(miss_method = self.miss_method, index_case = self.index_case, index_miss = self.index_miss, index_file = self.index_file, mode = 'embedding', sampletest = sampletest)
        self.mask = return_mask_of_data(data = self.data)
        self.array_embedding_size, self.array_embedding_location, self.mask_embed = return_array_size_and_location_for_columns_embedding(label_reverse = self.label_reverse, mask = self.mask, column_location = self.column_location)
        self.para = parameters_setting(model_name = 'VAE_embedding', data_shape = self.data.shape[1], data_embed_shape = self.array_embedding_size, epoch = epoch)
        self.cross_validation, self.loss_mode, self.kl_loss_mode, self.loss_axis, self.train_with_complete, self.noise_zero, self.test_iteration, self.epoch, self.alpha, self.loss_balance, self.noise_high_limit, self.learning_rate, self.model_structure = self.para.return_parameters()
        
        self.dsn = data_shuffle_noise(mode = 'embedding', noise_zero = self.noise_zero, high = self.noise_high_limit)
        self.model_estimate = Model_test(label_reverse = self.label_reverse, label_ori = self.label_ori, column_location = self.column_location, column_name = self.column_name, mode = 'embedding')
        self.ps = Performance_store(miss_method = self.miss_method, index_case = self.index_case, index_miss = self.index_miss, index_file = self.index_file, label_reverse = self.label_reverse, label_ori = self.label_ori, column_location = self.column_location, column_name = self.column_name, name = 'VAE_embedding', mode = 'embedding', index_pick = self.index_pick)
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
                output = return_layer(layer_input = inputs, output_size = self.network_latent_layer_D[0], activation = 'tanh')
            else:
                for i in range(len(self.network_latent_layer_D)):
                    if i == 0:
                        output = return_layer(layer_input = inputs, output_size = self.network_latent_layer_D[i], activation = 'tanh')
                    elif i == len(self.network_latent_layer_D) - 1:
                        output = return_layer(layer_input = output, output_size = self.network_latent_layer_D[i], activation = 'tanh')
                    else:
                        output = return_layer(layer_input = output, output_size = self.network_latent_layer_D[i], activation = 'tanh')
     
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
        if self.loss_mode == 'log_mse_masked':
            prob = distribution.log_prob(x)
            for index, i in enumerate(self.column_location):
                if self.label_reverse[index][0] == 'con':
                    if index == 0:
                        loss = tf.reduce_sum((prob[:, 0:i] * m[:, 0:i] - x[:, 0:i] * m[:, 0:i])**2)
                    else:
                        loss = loss + tf.reduce_sum((prob[:, self.column_location[index - 1]:i] * m[:, self.column_location[index - 1]:i] - x[:, self.column_location[index - 1]:i] * m[:, self.column_location[index - 1]:i])**2)
                else:
                    if index == 0:
                        loss_target = -(x[:, 0:i] + 1.)/2. * m[:, 0:i] * tf.log((prob[:, 0:i] + 1.)/2. + 1e-8)
                        loss = self.loss_balance * tf.reduce_sum(loss_target)
                    else:
                        loss_target = -(x[:, self.column_location[index - 1]:i] + 1.)/2. * m[:, self.column_location[index - 1]:i] * tf.log((prob[:, self.column_location[index - 1]:i] + 1.)/2. + 1e-8)
                        loss = loss + self.loss_balance * tf.reduce_sum(loss_target)
            loss = loss/(tf.reduce_sum(m) + 1e-8)
            
        elif self.loss_mode == 'log_mse':
            prob = distribution.log_prob(x)
            for index, i in enumerate(self.column_location):
                if self.label_reverse[index][0] == 'con':
                    if index == 0:
                        loss = tf.reduce_sum((prob[:, 0:i] - x[:, 0:i])**2)
                    else:
                        loss = loss + tf.reduce_sum((prob[:, self.column_location[index - 1]:i] - x[:, self.column_location[index - 1]:i])**2)
                else:
                    if index == 0:
                        loss_target = -(x[:, 0:i] + 1.)/2. * tf.log((prob[:, 0:i] + 1.)/2. + 1e-8)
                        loss = self.loss_balance * tf.reduce_sum(loss_target)
                    else:
                        loss_target = -(x[:, self.column_location[index - 1]:i] + 1.)/2. * tf.log((prob[:, self.column_location[index - 1]:i] + 1.)/2. + 1e-8)
                        loss = loss + self.loss_balance * tf.reduce_sum(loss_target)
            loss = loss/(tf.reduce_sum(m) + tf.reduce_sum(1. - m) + 1e-8)
        elif self.loss_mode == 'log_masked':
            prob = distribution.log_prob(x)
            loss = -tf.reduce_sum(prob * m, axis = self.loss_axis)/tf.reduce_sum(m, axis = self.loss_axis)
        elif self.loss_mode == 'log':
            prob = distribution.log_prob(x)
            loss = -tf.reduce_mean(prob, axis = self.loss_axis)
            
        return loss

    def kl_loss(self, output_from_generative, p_z = None, q_z = None):
        if self.kl_loss_mode == 'sample':
            loss = -1/2. * tf.reduce_mean(1. + output_from_generative[:, self.latent_size:] - output_from_generative[:, :self.latent_size]**2 + tf.math.exp(output_from_generative[:, self.latent_size:]))
        else:
            loss = tf.reduce_mean(tfp.distributions.kl_divergence(p_z, q_z), axis = self.loss_axis)
        return loss
        
        
    def return_cos_similarity(self, x, label, mode = 'tanh'):
        normalized_x, _ = tf.linalg.normalize(x, axis = 1)
        normalized_label, _ = tf.linalg.normalize(label[0], axis = 1)
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
        x, m, m_embed, n = self.para.return_placeholder()
        embedding_trainable_variables = []
        model_list = []

        for index, i in enumerate(self.column_location):
            if self.label_reverse[index][0] == 'con':
                if index == 0:
                    model_list.append(0)
                    x_i = x[:, 0:i]
                else:
                    x_i = tf.concat(values = [x_i, x[:, self.column_location[index - 1]:i]], axis = 1)
                    model_list.append(0)
            else:
                if index == 0:
                    model_i = return_embedding_model_for_columns(column_label_reverse = self.label_reverse[index])
                    model_list.append(model_i)
                    model_input = tf.cast(tf.math.argmax(x[:, 0:i], axis = 1), tf.float32)
                    x_i = model_i(model_input)
                    embedding_trainable_variables = embedding_trainable_variables + model_i.trainable_weights
                else:
                    model_i = return_embedding_model_for_columns(column_label_reverse = self.label_reverse[index])
                    model_list.append(model_i)
                    model_input = tf.cast(tf.math.argmax(x[:, self.column_location[index - 1]:i], axis = 1), tf.float32)
                    x_i = tf.concat(values = [x_i, model_i(model_input)], axis = 1) 
                    embedding_trainable_variables = embedding_trainable_variables + model_i.trainable_weights
        
        model_e = self.return_encode_network()
        gen_p = model_e(x_i * m_embed + n * (1. - m_embed))
        distribution_p = tfp.distributions.Normal(loc=gen_p[:, :self.latent_size], scale=tf.exp(gen_p[:, self.latent_size:]))
        model_d = self.return_decode_network()
        gen_x = model_d(distribution_p.sample())

        distribution_q = tfp.distributions.Normal(loc=np.zeros(self.latent_size, dtype=np.float32), scale=np.ones(self.latent_size, dtype=np.float32))
        if self.model_structure != 'sample':
            for index, i in enumerate(self.array_embedding_location):
                if self.label_reverse[index][0] == 'con':
                    if index == 0:
                        gen_x_embedding_reverse = gen_x[:, 0:i]
                    else:
                        gen_x_embedding_reverse = tf.concat(values = [gen_x_embedding_reverse, gen_x[:, self.array_embedding_location[index - 1]:i]], axis = 1)
                else:
                    if index == 0:
                        model_weights = model_list[index].trainable_weights
                        cos_similarity = self.return_cos_similarity(x = gen_x[:, 0:i], label = model_weights, mode = 'non') 
                        gen_x_embedding_reverse = cos_similarity
                    else:
                        model_weights = model_list[index].trainable_weights
                        cos_similarity = self.return_cos_similarity(x = gen_x[:, self.array_embedding_location[index - 1]:i], label = model_weights, mode = 'non')
                        gen_x_embedding_reverse = tf.concat(values = [gen_x_embedding_reverse, cos_similarity], axis = 1)
                        
            loss_p2 = self.kl_loss(output_from_generative = gen_p, p_z = distribution_p, q_z = distribution_q)
            model_ddm = self.return_decode_network_divide(output_mode = 'mean')
            model_ddv = self.return_decode_network_divide(output_mode = 'var')
            
            gen_x_embedding_reverse_mean = model_ddm(gen_x_embedding_reverse)
            gen_x_embedding_reverse_var = model_ddv(gen_x_embedding_reverse)
            
            distribution = tfp.distributions.Normal(loc=gen_x_embedding_reverse_mean, scale=tf.exp(gen_x_embedding_reverse_var))
            
            loss_p1 = self.loss(distribution = distribution, x = x, m = m)
            loss = tf.reduce_sum(loss_p1 + self.alpha * loss_p2)
            #loss = tf.reduce_sum(loss_p1)
            solver = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(loss, var_list = model_e.trainable_variables + embedding_trainable_variables + model_d.trainable_variables + model_ddm.trainable_variables + model_ddv.trainable_variables)
            return loss, loss_p1, loss_p2, solver, gen_x_embedding_reverse_mean, x, m, m_embed, n
        else:
            loss_p2 = self.kl_loss(output_from_generative = gen_p, p_z = distribution_p, q_z = distribution_q)
            model_ddm = self.return_decode_network_divide(output_mode = 'mean')
            model_ddv = self.return_decode_network_divide(output_mode = 'var')
            
            gen_x_embedding_reverse_mean = model_ddm(gen_x)
            gen_x_embedding_reverse_var = model_ddv(gen_x)
            
            distribution = tfp.distributions.Normal(loc=gen_x_embedding_reverse_mean, scale=tf.exp(gen_x_embedding_reverse_var))
            
            loss_p1 = self.loss(distribution = distribution, x = x, m = m)
            loss = tf.reduce_sum(loss_p1 + self.alpha * loss_p2)
            #loss = tf.reduce_sum(self.alpha * loss_p1)
            solver = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(loss, var_list = model_e.trainable_variables + embedding_trainable_variables + model_d.trainable_variables + model_ddm.trainable_variables + model_ddv.trainable_variables)
            return loss, loss_p1, loss_p2, solver, gen_x_embedding_reverse_mean, x, m, m_embed, n
   
    
    def return_Kfold(self, tr, te):
        data = self.data.copy()
        mask = self.mask.copy()
        mask_embed = self.mask_embed.copy()
        df_full = self.df_original.copy()
        df_miss = self.df_miss.copy()
            
        train_d = data[tr]
        test_d = data[te]
        train_m = mask[tr]
        test_m = mask[te]
        train_m_embed = mask_embed[tr]
        test_m_embed = mask_embed[te]
        df_train_label = df_full.iloc[tr].copy().reset_index(drop = True)
        df_test_label = df_full.iloc[te].copy().reset_index(drop = True)
        df_train_miss = df_miss.iloc[tr].copy().reset_index(drop = True)
        df_test_miss = df_miss.iloc[te].copy().reset_index(drop = True)
        return train_d, train_m, train_m_embed, df_train_label, df_train_miss, test_d, test_m, test_m_embed, df_test_label, df_test_miss

    def train_process(self):
        if self.cross_validation is None:
            loss, loss_p1, loss_p2, solver, gen_x, x, m, m_embed, n = self.return_defined_network_for_mode()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            if self.train_with_complete:
                train_data, train_mask, train_mask_embed = self.dsn.dataNonenan(data = self.data.copy(), mask = self.mask.copy(), mask_embed = self.mask_embed.copy())
            else:
                train_data, train_mask, train_mask_embed = self.data, self.mask, self.mask_embed

            for epoc in range(self.epoch):
                train_d, train_m, train_m_embed = self.dsn.data_shuffle(train_data, train_mask, train_mask_embed)
                for iteration in range(int(train_d.shape[0]/self.batch_size)):
                    train_d_batch = train_d[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    train_m_batch = train_m[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    train_m_embed_batch = train_m_embed[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    noise_batch = self.dsn.return_noise_batch(shape = [self.batch_size, m_embed.shape[1]])
                    
                    _ = sess.run([solver], feed_dict = {x: train_d_batch, m: train_m_batch, m_embed: train_m_embed_batch, n: noise_batch})
                    
            data_noised = self.dsn._add_noise_(train_data = self.data.copy(), train_mask = self.mask.copy())
            data_imputed = self.testIteration(sess = sess, gen_x = gen_x, data = data_noised, mask = self.mask.copy(), mask_embed = self.mask_embed.copy(), x = x, m = m, m_embed = m_embed, n = n)
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
                train_d, train_m, train_m_embed, df_train_label, df_train_miss, test_d, test_m, test_m_embed, df_test_label, df_test_miss = self.return_Kfold(tr, te)
                loss, loss_p1, loss_p2, solver, gen_x, x, m, m_embed, n = self.return_defined_network_for_mode()###
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
                if self.train_with_complete:
                    train_d, train_m, train_m_embed = self.dsn.dataNonenan(data = train_d.copy(), mask = train_m.copy(), mask_embed = train_m_embed.copy())
                else:
                    train_d, train_m, train_m_embed = train_d, train_m, train_m_embed

                for epoc in range(self.epoch):
                    train_d_, train_m_, train_m_embed_ = self.dsn.data_shuffle(train_data = train_d, train_mask = train_m, train_mask_embed = train_m_embed)
                    for iteration in range(int(train_d.shape[0]/self.batch_size)):
                        train_d_batch = train_d_[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                        train_m_batch = train_m_[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                        train_m_embed_batch = train_m_embed_[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                        noise_batch = self.dsn.return_noise_batch(shape = [self.batch_size, m_embed.shape[1]])
                    
                        _ = sess.run([solver], feed_dict = {x: train_d_batch, m: train_m_batch, m_embed: train_m_embed_batch, n: noise_batch})
                                   
                test_noise = self.dsn._add_noise_(train_data = test_d, train_mask = test_m)
                noise_batch_full = self.dsn.return_noise_batch(shape = test_noise.shape)
                test_d_imputed = self.testIteration(sess = sess, gen_x = gen_x, data = test_noise, mask = test_m.copy(), mask_embed = test_m_embed.copy(), x = x, m = m, m_embed = m_embed, n = n)
                test_d = test_d_imputed * (1. - test_m) + test_noise * test_m
                data_list.append(test_d)
                mask_list.append(test_m)
                df_full_list.append(df_test_label)
                df_miss_list.append(df_test_miss)
                
                sess.close()
                tf.keras.backend.clear_session()
                        
            data_c, mask_c, df_original_c, df_miss_c = self.model_estimate.cross_validation_result(data_list, mask_list, df_full_list, df_miss_list)
            con_loss, cat_accuracy = self.model_estimate.model_test(data = data_c, mask = mask_c, df_original = df_original_c)
            self.ps.storeImputed(data = data_c, df_original = df_original_c, df_miss = df_miss_c)
            return con_loss, cat_accuracy
        
        
    def train_process_sample(self):
        #print(self.index_case)
        if self.cross_validation is None:
            loss, loss_p1, loss_p2, solver, gen_x, x, m, m_embed, n = self.return_defined_network_for_mode()
            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
            
            con_list, cat_list = [], []
            
            if self.train_with_complete:
                train_data, train_mask, train_mask_embed = self.dsn.dataNonenan(data = self.data.copy(), mask = self.mask.copy(), mask_embed = self.mask_embed.copy())
            else:
                train_data, train_mask, train_mask_embed = self.data, self.mask, self.mask_embed

            for epoc in range(self.epoch):
                train_d, train_m, train_m_embed = self.dsn.data_shuffle(train_data, train_mask, train_mask_embed)
                for iteration in range(int(train_d.shape[0]/self.batch_size)):
                    train_d_batch = train_d[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    train_m_batch = train_m[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    train_m_embed_batch = train_m_embed[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                    noise_batch = self.dsn.return_noise_batch(shape = [self.batch_size, m_embed.shape[1]])
                    
                    _ = sess.run([solver], feed_dict = {x: train_d_batch, m: train_m_batch, m_embed: train_m_embed_batch, n: noise_batch})
                    
                data_noised = self.dsn._add_noise_(train_data = self.data.copy(), train_mask = self.mask.copy())
                data_imputed = self.testIteration(sess = sess, gen_x = gen_x, data = data_noised, mask = self.mask.copy(), mask_embed = self.mask_embed.copy(), x = x, m = m, m_embed = m_embed, n = n)
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
                train_d, train_m, train_m_embed, df_train_label, df_train_miss, test_d, test_m, test_m_embed, df_test_label, df_test_miss = self.return_Kfold(tr, te)
                loss, loss_p1, loss_p2, solver, gen_x, x, m, m_embed, n = self.return_defined_network_for_mode()###
                sess = tf.Session()
                sess.run(tf.global_variables_initializer())
                
                con_list, cat_list = [], []
                
                if self.train_with_complete:
                    train_d, train_m, train_m_embed = self.dsn.dataNonenan(data = train_d.copy(), mask = train_m.copy(), mask_embed = train_m_embed.copy())
                else:
                    train_d, train_m, train_m_embed = train_d, train_m, train_m_embed

                for epoc in range(self.epoch):
                    train_d_, train_m_, train_m_embed_ = self.dsn.data_shuffle(train_data = train_d, train_mask = train_m, train_mask_embed = train_m_embed)
                    for iteration in range(int(train_d.shape[0]/self.batch_size)):
                        train_d_batch = train_d_[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                        train_m_batch = train_m_[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                        train_m_embed_batch = train_m_embed_[iteration * self.batch_size : (iteration + 1) * self.batch_size]
                        noise_batch = self.dsn.return_noise_batch(shape = [self.batch_size, m_embed.shape[1]])
                    
                        _ = sess.run([solver], feed_dict = {x: train_d_batch, m: train_m_batch, m_embed: train_m_embed_batch, n: noise_batch})
                                   
                    test_noise = self.dsn._add_noise_(train_data = test_d, train_mask = test_m)
                    noise_batch_full = self.dsn.return_noise_batch(shape = test_noise.shape)
                    test_d_imputed = self.testIteration(sess = sess, gen_x = gen_x, data = test_noise, mask = test_m.copy(), mask_embed = test_m_embed.copy(), x = x, m = m, m_embed = m_embed, n = n)
                    test_d = test_d_imputed * (1. - test_m) + test_noise * test_m
                    con_loss, cat_accuracy = self.model_estimate.model_test(data = test_d, mask = test_m, df_original = df_test_label)
                    con_list.append(con_loss)
                    cat_list.append(cat_accuracy)
                index_re = self.ps.return_con_cat_index(con_list = con_list, cat_accuracy = cat_list)  
                index_re_list.append(index_re)
                sess.close()
                tf.keras.backend.clear_session()

            return np.around(np.mean(index_re_list))
        

    def testIteration(self, sess, gen_x, data = None, mask = None, mask_embed = None, x = None, m = None, m_embed = None, n = None):
        if self.test_iteration is None:
            noise_batch = self.dsn.return_noise_batch(shape = [data.shape[0], m_embed.shape[1]])
            generate_data = sess.run(gen_x, feed_dict = {x: data, m: mask, m_embed: mask_embed, n: noise_batch})
        else:
            generate_data = data
            for i in range(self.test_iteration):
                noise_batch = self.dsn.return_noise_batch(shape = [data.shape[0], m_embed.shape[1]])
                generate_data = sess.run(gen_x, feed_dict = {x: generate_data, m: mask, m_embed: mask_embed, n: noise_batch})
                
        return generate_data
                
    
    