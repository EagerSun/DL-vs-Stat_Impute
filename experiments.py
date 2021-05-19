from GAIN import GAIN
from VAE import VAE
from GAIN_embedding import GAIN_embedding
from VAE_embedding import VAE_embedding
from utils.returnMedianrange import returnMedianrange
import gc
import numpy as np

class experiments():
    def __init__(self, model_name):
        self.model_name = model_name
       
        self.miss_methods = ["MCAR", "MAR", "MNAR"]
        self.case_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]# list for case indices, you could new indices in this list for new datasets
        self.batches = {1:32, 2:64, 3:256, 4:32, 5:256, 6:128, 7:512, 8:32, 9:64, 10:512}# Dictionary for batch-size of different datasets, querying case index, you could new indices by {case_index}: batch_size in this dictionary for new datasets
        ## Dictionary for measuring method of epoch size for different datasets, querying case index.
        ## In detail, if you want to measure the epoch size from performance of continuous features, mark {case_index}: "continuous_first" in the dictionary, 
        ## else mark {case_index}: "categorical_first"
        self.index_picks = {1:"continuous_first", 2:"continuous_first", 3:"continuous_first", 4:"categorical_first", 5:"categorical_first", 6:"continuous_first", 7:"continuous_first", 8:"continuous_first", 9:"continuous_first", 10:"continuous_first"}
        
        self.miss_list = [10, 30, 50]
        self.num_sampletest = 10 # num of sample test defined in ./Rcode/sample_miss.R
        self.num_experiments = 100 # num of training test defined in ./Rcode/sample_miss.R
        self.Epoch_sampletest = 500 # num of epoch during each sample test case
        
    def return_model_case(self):
        if self.model_name == 'GAIN':
            case_list = self.case_list
        elif self.model_name == 'VAE':
            case_list = self.case_list
        elif self.model_name == 'GAIN_embedding':
            case_list = [4, 5, 6, 7, 8, 9, 10]# If the new dataset have categorical features add the case index in this list
        elif self.model_name == 'VAE_embedding':
            case_list = [4, 5, 6, 7, 8, 9, 10]# If the new dataset have categorical features add the case index in this list
            
        return case_list 
    
    def return_model(self, miss_method, index_case, index_miss, index_file, batch_num, epoch, sampletest, index_pick):
        if self.model_name == 'GAIN':
            model = GAIN(miss_method = miss_method, index_case = index_case, index_miss = index_miss, index_file = index_file, batch_num = batch_num, epoch = epoch, sampletest = sampletest, index_pick = index_pick)
        elif self.model_name == 'VAE':
            model = VAE(miss_method = miss_method, index_case = index_case, index_miss = index_miss, index_file = index_file, batch_num = batch_num, epoch = epoch, sampletest = sampletest, index_pick = index_pick)
        elif self.model_name == 'GAIN_embedding':
            model = GAIN_embedding(miss_method = miss_method, index_case = index_case, index_miss = index_miss, index_file = index_file, batch_num = batch_num, epoch = epoch, sampletest = sampletest, index_pick = index_pick)
        elif self.model_name == 'VAE_embedding':
            model = VAE_embedding(miss_method = miss_method, index_case = index_case, index_miss = index_miss, index_file = index_file, batch_num = batch_num, epoch = epoch, sampletest = sampletest, index_pick = index_pick)
            
        return model
            
    def run_model(self):
        epoch_dict = self.run_sampletest()
        self.run_experiment(epoch_dict = epoch_dict)
        
    def run_sampletest(self):
        epoch_case = {}
        case_list = self.return_model_case()
        for miss_method in self.miss_methods:
            epoch_case[miss_method] = {}
            for index_case in case_list:
                #print("case: " + str(index_case))
                epoch_case[miss_method][index_case] = {}
                for index_miss in self.miss_list:
                    #print("miss: " + str(index_miss))
                    epoch_stop = []
                    for index_file in range(self.num_sampletest): 
                        #print(index_case, index_miss, index_file)
                        model = self.return_model(miss_method = miss_method, index_case = index_case, index_miss = index_miss, index_file = index_file, batch_num = self.batches[index_case], epoch = self.Epoch_sampletest, sampletest = True, index_pick = self.index_picks[index_case])
                        index_stop = model.train_process_sample()
                        #print(index_stop)
                        epoch_stop.append(index_stop)
                    epoch_case[miss_method][index_case][index_miss] = int(np.around(np.mean(epoch_stop)))
                    del epoch_stop
                    gc.collect()
        return epoch_case
    
    def run_experiment(self, epoch_dict):
        case_list = self.return_model_case()

        for miss_method in self.miss_methods:
            for index_case in case_list:
                for index_miss in self.miss_list:  
                    con_loss_last = []
                    cat_accuracy_last = []
                    for index_file in range(self.num_experiments): 
                        model = self.return_model(miss_method = miss_method, index_case = index_case, index_miss = index_miss, index_file = index_file, batch_num = self.batches[index_case], epoch = epoch_dict[miss_method][index_case][index_miss], sampletest = False, index_pick = self.index_picks[index_case])
                        con, cat = model.train_process()
                        con_loss_last.append(con)
                        cat_accuracy_last.append(cat)

                    returnMedianrange(con_loss_last, cat_accuracy_last, self.num_experiments, miss_method, index_case, index_miss, self.model_name)
                    del con_loss_last, cat_accuracy_last
                    gc.collect()
            
        
        
        
