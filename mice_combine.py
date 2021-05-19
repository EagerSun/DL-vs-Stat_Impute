import os
import pandas as pd
import statistics
import gc
from miceCombine.return_average import return_average_mice

index_case_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
index_miss_list = [10, 30, 50]
miss_methods = ["MCAR", "MAR", "MNAR"]
num_of_test = 100
for miss_method in miss_methods:
    for index_case in index_case_list:
        for index_miss in index_miss_list:  
            for index_file in range(num_of_test):
                try:
                    return_average_mice(miss_method, index_case, index_miss, index_file)
                except:
                    continue
gc.collect()