import numpy as np
import tensorflow as tf

class parameters_setting():
    def __init__(self, model_name, data_shape, data_embed_shape = None, epoch = 500):
        '''
        Common parameters for all deep learning models
        '''
        self.model = model_name # model's name, e.g., GAIN, VAE, VAE_embedding, GAIN_embedding
        self.data_shape = data_shape # Input data shape
        self.data_embed_shape = data_embed_shape # Embedded data shape
        self.model_structure = 'sample' # specific parameter for embedding model
        self.cross_validation = None # Whether to do the cross-validation or not
        self.epoch = epoch # Num of epochs
        
    def return_parameters(self):
        if self.model == 'GAIN':
            return self.return_gain_parameters()
        elif self.model == 'GAIN_embedding':
            return self.return_gain_embed_parameters()
        elif self.model == 'VAE':
            return self.return_vae_parameters()
        elif self.model == 'VAE_embedding':
            return self.return_vae_embed_parameters()
        
    def return_layer_size(self):
        if self.model == 'GAIN':
            return self.return_gain_layer_size()
        elif self.model == 'GAIN_embedding':
            return self.return_gain_embed_layer_size()
        elif self.model == 'VAE':
            return self.return_vae_layer_size()
        elif self.model == 'VAE_embedding':
            return self.return_vae_embed_layer_size()
        
    def return_placeholder(self):
        if self.model == 'GAIN':
            return self.return_gain_placeholder()
        elif self.model == 'GAIN_embedding':
            return self.return_gain_embed_placeholder()
        elif self.model == 'VAE':
            return self.return_vae_placeholder()
        elif self.model == 'VAE_embedding':
            return self.return_vae_embed_placeholder()
        
        
###############################################         GAIN          ###################################################################   

    def return_gain_parameters(self):
        '''
        Define the GAIN's structure info
        '''
        cross_validation = self.cross_validation # whether to do cross-validation
        # L_g = l + alpha * c_g 
        loss_mode = 'mse_masked' # detail loss calculation for l
        d_loss_mode = 'log_masked' #c_d
        g_loss_mode = 'log_masked' #c_g
        epoch = self.epoch #num of epochs
        alpha = 10. # alpha as balance parameters
        loss_balance =1.0
        p_hint = 0.8
        noise_high_limit = 1e-1 # noise range
        '''
        Define the GAIN's layers for G and D.
        '''
        network_layer_G = [self.data_shape * 2, self.data_shape * 1, self.data_shape * 1]
        network_layer_D = [self.data_shape * 2, self.data_shape * 1, self.data_shape * 1]
        
        return cross_validation, loss_mode, d_loss_mode, g_loss_mode, epoch, alpha, loss_balance, p_hint, noise_high_limit
    
    def return_gain_layer_size(self):
        network_layer_G = [self.data_shape * 2, self.data_shape * 1, self.data_shape * 1]
        network_layer_D = [self.data_shape * 2, self.data_shape * 1, self.data_shape * 1]
        return network_layer_G, network_layer_D
    
    def return_gain_placeholder(self):
        x = tf.placeholder(tf.float32, shape = [None, self.data_shape])
        m = tf.placeholder(tf.float32, shape = [None, self.data_shape])
        h = tf.placeholder(tf.float32, shape = [None, self.data_shape])
        return x, m, h

##########################################        GAIN_embedding     #################################################################### 
    
    def return_gain_embed_parameters(self):
        cross_validation = self.cross_validation # whether to do cross-validation
        # L_g = l + alpha * c_g 
        epoch = self.epoch # num of epochs
        alpha = 10. # alpha
        loss_balance = 1.0
        p_hint = 0.8
        noise_high_limit = 1e-1 # noise range
        '''
        Define the GAIN's layers for G and D.
        '''
        loss_mode = 'mse_masked' # detail loss calculation for l
        d_loss_mode = 'log_masked' #c_d
        g_loss_mode = 'log_masked' #c_g
        model_structure = self.model_structure
        
        return cross_validation, loss_mode, d_loss_mode, g_loss_mode, epoch, alpha, loss_balance, p_hint, noise_high_limit, model_structure
    
    def return_gain_embed_layer_size(self):
        if self.model_structure != 'sample':
            network_layer_G = [self.data_embed_shape * 2, self.data_embed_shape * 1, self.data_embed_shape * 1]
            network_layer_D = [self.data_shape * 2, self.data_shape * 1, self.data_shape * 1]
        else:
            network_layer_G = [self.data_embed_shape * 2, self.data_embed_shape * 1, self.data_shape * 1]
            network_layer_D = [self.data_shape * 2, self.data_shape * 1, self.data_shape * 1]
            
        return network_layer_G, network_layer_D
    
    def return_gain_embed_placeholder(self):
        x = tf.placeholder(tf.float32, shape = [None, self.data_shape])
        m = tf.placeholder(tf.float32, shape = [None, self.data_shape])
        m_embed = tf.placeholder(tf.float32, shape = [None, self.data_embed_shape])     
        h = tf.placeholder(tf.float32, shape = [None, self.data_shape])
        n = tf.placeholder(tf.float32, shape = [None, self.data_embed_shape])
        return x, m, m_embed, h, n
    
###############################################          VAE          ################################################################### 

    def return_vae_parameters(self):
        '''
        Define the VAE's structure info
        '''
        cross_validation = self.cross_validation # whether to do the cross-validation
        #L = L_re + alpha * kl
        loss_mode = 'log_masked' # L_re
        kl_loss_mode = 'complex' # kl
        loss_axis = None
        train_with_complete = False # whether train VAE with full portion of missing dataset
        test_iteration = 20 # num of iterative process after former training
        
        epoch = self.epoch # num of epoch
        alpha = 1e-2 # alpha
        
        loss_balance = 1.0
        noise_high_limit = 1e-1 # noise range
        noise_zero = True
        learning_rate = 1e-3 # learning rate 
        return cross_validation, loss_mode, kl_loss_mode, loss_axis, train_with_complete, noise_zero, test_iteration, epoch, alpha, loss_balance, noise_high_limit, learning_rate
    
    def return_vae_layer_size(self):
        return self.return_vae_layers()
        
    def return_vae_layers(self):
        if int(self.data_shape/4.) < 4:
            latent_size = 4
            mid_layer = int((latent_size + self.data_shape)/2.)
        else:
            latent_size = int(self.data_shape/4.)
            mid_layer = int(self.data_shape/2.)
            
        network_layer_G = [self.data_shape, mid_layer, latent_size]
        network_latent_layer_G = [latent_size]
        network_layer_D = [latent_size, mid_layer, self.data_shape]
        network_latent_layer_D = [self.data_shape]
        
        return latent_size, network_layer_G, network_latent_layer_G, network_layer_D, network_latent_layer_D 
    
    def return_vae_placeholder(self):
        x = tf.placeholder(tf.float32, shape = [None, self.data_shape])
        m = tf.placeholder(tf.float32, shape = [None, self.data_shape])
        return x, m
    
##########################################        VAE_embedding     #################################################################### 

    def return_vae_embed_parameters(self):
        '''
        Define the VAE's structure info
        '''
        cross_validation = self.cross_validation # whether to do the cross-validation
        #L = L_re + alpha * kl
        loss_mode = 'log_masked' # L_re
        kl_loss_mode = 'complex' # kl
        loss_axis = None
        train_with_complete = False # whether train VAE with full portion of missing dataset
        test_iteration = 20 # num of iterative process after former training
        
        epoch = self.epoch # num of epoch
        alpha = 1e-2 # alpha

        loss_balance = 1.0
        noise_high_limit = 1e-1 # noise range
        noise_zero = True
        learning_rate = 1e-3 # learning rate
        model_structure = self.model_structure
        return cross_validation, loss_mode, kl_loss_mode, loss_axis, train_with_complete, noise_zero, test_iteration, epoch, alpha, loss_balance, noise_high_limit, learning_rate, model_structure
    
    def return_vae_embed_layer_size(self):
        if self.model_structure != 'sample':
            return self.return_vae_embed_layers()
        else:
            return self.return_vae_embed_layers()
        
    def return_vae_embed_layers(self):
        if int(self.data_embed_shape/4.) < 4:
            latent_size = 4
            mid_layer = int((latent_size + self.data_embed_shape)/2.)
        else:
            latent_size = int(self.data_embed_shape/4.)
            mid_layer = int(self.data_embed_shape/2.)
            
        network_layer_G = [self.data_embed_shape, mid_layer, latent_size]
        network_latent_layer_G = [latent_size]
        if self.model_structure != 'sample':
            network_layer_D = [latent_size, mid_layer, self.data_embed_shape, self.data_embed_shape]
            network_latent_layer_D = [self.data_shape, self.data_shape]
        else:
            network_layer_D = [latent_size, mid_layer, self.data_shape, self.data_shape]
            network_latent_layer_D = [self.data_shape]
        return latent_size, network_layer_G, network_latent_layer_G, network_layer_D, network_latent_layer_D
    
    def return_vae_embed_placeholder(self):
        x = tf.placeholder(tf.float32, shape = [None, self.data_shape])
        m = tf.placeholder(tf.float32, shape = [None, self.data_shape])
        m_embed = tf.placeholder(tf.float32, shape = [None, self.data_embed_shape])
        n = tf.placeholder(tf.float32, shape = [None, self.data_embed_shape])
        return x, m, m_embed, n
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
