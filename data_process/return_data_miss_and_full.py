from data_process.return_data_miss_and_full_train import return_data_miss_and_full_train
from data_process.return_data_miss_and_full_sampletest import return_data_miss_and_full_sampletest

def return_data_miss_and_full(miss_method, index_case, index_miss, index_file, mode = 'one_hot', sampletest = False):
    if sampletest:
        return return_data_miss_and_full_sampletest(miss_method = miss_method, index_case = index_case, index_miss = index_miss, index_file = index_file, mode = mode)
    
    else:
        return return_data_miss_and_full_train(miss_method = miss_method, index_case = index_case, index_miss = index_miss, index_file = index_file, mode = mode)