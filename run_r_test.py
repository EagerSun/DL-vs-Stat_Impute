from Rimputed_test import Rimputed_test
import numpy as np
import pandas as pd
import gc

index_case_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]# list for case indices, you could new indices in this list for new datasets
index_miss_list = [10, 30, 50]
num_of_test = 100
miss_methods = ["MCAR", "MNAR", "MAR"]
imputed_method_list = ["MICE", "MISS_FOREST"] 
for miss_method in miss_methods:
    for imputed_method in imputed_method_list:
        for index_case in index_case_list:
            for index_miss in index_miss_list:  
                Rimputed_test(miss_method = miss_method, index_case = index_case, index_miss = index_miss, imputed_method = imputed_method, num_of_test = num_of_test)
gc.collect()
