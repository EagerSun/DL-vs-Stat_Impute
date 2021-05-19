from confusionmatrix import confusionmatrix
from return_data import return_data
import scipy.stats as stats
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#data, df_full = return_data(index_case = 5, index_miss = 20, index_file = 1)

def value_calc(d1, d2, mode, df1 = None, df2 = None):
    list1, list2 = d1, d2
    #print(len(list1), len(list2))
    if mode == 'concon':
        matrix = np.corrcoef(list1, list2)
        value = matrix[0][1]
    elif mode == 'concat':
        #value, _ = stats.pointbiserialr(list1, list2)
        matrix = np.corrcoef(list1, list2)
        value = matrix[0][1]
    else:
        confusionmatrix = pd.crosstab(df1, df2)
        confusionmatrix = confusionmatrix.to_numpy()
        chi2 = stats.chi2_contingency(confusionmatrix)[0]
        n = np.sum(confusionmatrix)
        phi2 = chi2/n
        #print(confusionmatrix.shape)
        r,k = confusionmatrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        value = np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    return value
        

def coefficient_calculate(index_case):
    data, df_full = return_data(index_case = index_case)
    columns = df_full.columns
    con_cat = [i[:3] for i in columns]
    matrix = np.zeros((len(columns), len(columns)), dtype = np.float32)
    for i in range(len(columns)):
        for j in range(i, len(columns)):
            concat = [con_cat[i], con_cat[j]]
            if 'con' in concat and 'cat' in concat:
                if (con_cat[i] == 'cat' and len(set(df_full[columns[i]].to_list())) == 2) or (con_cat[j] == 'cat' and len(set(df_full[columns[j]].to_list())) == 2):
                    if con_cat[i] == 'cat':
                        value = value_calc(list(data[:, i]), df_full[columns[j]].to_list(), mode = 'concat')
                    else:
                        value = value_calc(df_full[columns[i]].to_list(), list(data[:, j]), mode = 'concat')
                elif con_cat[i] == 'cat':
                    value = value_calc(list(data[:, i]), df_full[columns[j]].to_list(), mode = 'concon')
                elif con_cat[j] == 'cat':
                    value = value_calc(df_full[columns[i]].to_list(), list(data[:, j]), mode = 'concon')
            elif 'con' in concat and 'cat' not in concat:
                value = value_calc(df_full[columns[i]].to_list(), df_full[columns[j]].to_list(), mode = 'concon')
            else:
                value = value_calc(df_full[columns[i]].to_list(), df_full[columns[j]].to_list(), mode = 'catcat', df1 = df_full[columns[i]], df2 = df_full[columns[j]])
            matrix[i][j], matrix[j][i] = value, value
    return matrix, list(df_full.columns)

def plot_coefficient(index_case):
    matrix, labels = coefficient_calculate(index_case)
    fig, ax = plt.subplots(figsize = (10,10))
    #ax.set_title(name)
    im = ax.imshow(matrix, cmap=plt.cm.Blues, vmin = 0.0, vmax = 1.0)
    ax.set_ylim([-0.5, -0.5+len(labels)])
    if index_case != 5 and index_case != 7 and index_case != 2:
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticks(np.arange(len(labels)))

        # ... and label them with the respective list entries
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.15)
    fig.colorbar(im, cax = cax)
    mathpath = os.path.join(os.getcwd(), "Coefficient")
    if not os.path.exists(mathpath):
        os.mkdir(mathpath)
    imageaddress = os.path.join(mathpath, "case_{0}.png".format(index_case))
    #imageaddress = os.path.join(imagefolder, 'case_{1}_miss_{2}_columnname_{3}.png'.format(model, cindex, mindex, name))
    fig.savefig(imageaddress)
            
                
            