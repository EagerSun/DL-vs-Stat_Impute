import numpy as np

class data_shuffle_noise():
    def __init__(self, mode, noise_zero, high):
        self.mode = mode
        self.noise_zero = noise_zero
        self.high = high
        if self.mode == 'one_hot':
            self.low = 0.0
        else:
            self.low = -self.high
            
    def _add_noise_(self, train_data, train_mask):
        if self.noise_zero:
            train_data[np.isnan(train_data)] = 0.0
            return train_data
        else:
            noise = np.random.uniform(low = self.low, high = self.high, size = train_data.shape)
            train_data[np.isnan(train_data)] = 0.0
            return train_data * train_mask + noise * (1. - train_mask)
        
    def dataNonenan(self, data, mask, mask_embed):
        index = []
        for i in range(data.shape[0]):
            if str(np.sum(data[i, :])) != 'nan':
                index.append(i)
        print(data[index, :].shape[0])  
        if self.mode == 'one_hot':
            return data[index, :], mask[index, :]
        else:
            return data[index, :], mask[index, :], mask_embed[index, :]
    
    def return_noise_batch(self, shape):
        if self.noise_zero:
            return np.zeros(shape, dtype = np.float32)
        else:
            return np.random.uniform(low=-self.high, high=self.high, size=shape)

    def data_shuffle(self, train_data, train_mask, train_mask_embed = None):
        idx = np.arange(train_data.shape[0])
        np.random.shuffle(idx)
        if self.mode == 'one_hot':
            train_d = train_data[idx]
            train_m = train_mask[idx]
            train_d = self._add_noise_(train_data = train_d, train_mask = train_m)
            return train_d, train_m
        else:
            train_d = train_data[idx]
            train_m = train_mask[idx]
            train_m_embed = train_mask_embed[idx]
            train_d = self._add_noise_(train_data = train_d, train_mask = train_m)
            return train_d, train_m, train_m_embed

    
