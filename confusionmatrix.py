import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os
import matplotlib.pyplot as plt

def print_confusionmatrix(list1, list2, labels, name, missing, cindex, mindex, model):
    font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 10,
        }
    matrix = confusion_matrix(list1, list2, labels = labels)
    fig, ax = plt.subplots(figsize = (len(labels),len(labels)))
    ax.set_title(name, fontdict=font)
    #cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(matrix, cmap=plt.cm.Blues)
    ax.set_ylim([-0.5, -0.5+len(labels)])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticks(np.arange(len(labels)))

    # ... and label them with the respective list entries
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    ax.set_ylabel("True", fontdict=font)
    ax.set_xlabel("Predict", fontdict=font)
    maxlength = max([len(str(i)) for i in labels])
    if maxlength > 2:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            c = matrix[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
    #fig.colorbar(im)
    mathpath = os.path.join(os.getcwd(), "Image")
    if not os.path.exists(mathpath):
        os.mkdir(mathpath)
    misspatternpath = os.path.join(mathpath, missing)
    if not os.path.exists(misspatternpath):
        os.mkdir(misspatternpath)
    casepath = os.path.join(misspatternpath, "Case{0}".format(cindex))
    if not os.path.exists(casepath):
        os.mkdir(casepath)
    ratiopath = os.path.join(casepath, "miss{0}".format(mindex))
    if not os.path.exists(ratiopath):
        os.mkdir(ratiopath)
    modelpath = os.path.join(ratiopath, model)
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)
    imageaddress = os.path.join(modelpath, "columnname_{0}.png".format(name))
    fig.savefig(imageaddress, bbox_inches='tight')

def plot_continuousscatter(list1, list2, name, missing, cindex, mindex, model):
    font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 10,
        }
    vrange = max(list1) - min(list1)
    bias = float(1/10)*vrange
    vmin, vmax = min(list1)-bias, max(list1) + bias
    fig, ax = plt.subplots(figsize = (4,4))
    ax.scatter(list1, list2)
    ax.set_xlabel("True", fontdict=font)
    ax.set_ylabel("Predict", fontdict=font)
    ax.set_title(name, fontdict=font)
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])
    mathpath = os.path.join(os.getcwd(), "Image")
    if not os.path.exists(mathpath):
        os.mkdir(mathpath)
    misspatternpath = os.path.join(mathpath, missing)
    if not os.path.exists(misspatternpath):
        os.mkdir(misspatternpath)
    casepath = os.path.join(misspatternpath, "Case{0}".format(cindex))
    if not os.path.exists(casepath):
        os.mkdir(casepath)
    ratiopath = os.path.join(casepath, "miss{0}".format(mindex))
    if not os.path.exists(ratiopath):
        os.mkdir(ratiopath)
    modelpath = os.path.join(ratiopath, model)
    if not os.path.exists(modelpath):
        os.mkdir(modelpath)
    imageaddress = os.path.join(modelpath, "columnname_{0}.png".format(name))
    fig.savefig(imageaddress, bbox_inches='tight')
            
def confusionmatrix(cindex, mindex, model, missing, findex):
    if model not in ["MICE", "MISS_FOREST"]:
        mainpath = os.path.join(os.getcwd(), "data_stored", "data_imputed")
        misspath = os.path.join(mainpath, missing)
        casepath = os.path.join(misspath, "Case{0}".format(cindex))
        misspath = os.path.join(casepath, "miss{0}".format(mindex))
        modelpath = os.path.join(misspath, model)
        df_imputed = pd.read_csv( os.path.join(modelpath, "imputed", "{0}.csv").format(findex))
    else:
        if model == "MICE":
            mainpath = os.path.join(os.getcwd(), "data_stored", "data_mice")
        else:
            mainpath = os.path.join(os.getcwd(), "data_stored", "data_missforest")
        misspath = os.path.join(mainpath, missing)
        casepath = os.path.join(misspath, "Case{0}".format(cindex))
        misspath = os.path.join(casepath, "miss{0}".format(mindex))
        df_imputed = pd.read_csv( os.path.join(misspath, "{0}.csv").format(findex))
    
    df_miss = pd.read_csv(os.path.join(os.getcwd(), "data_stored", "data_miss", missing, "Case{0}".format(cindex), "miss{0}".format(mindex), "{0}.csv").format(findex))
    df_full = pd.read_csv(os.path.join(os.path.join(os.getcwd(), "data_stored", "data"), "{0}.csv".format(cindex)))
    columns = list(df_full.columns)
    for i in columns:
        list_m, list_f, list_i = df_miss[i].to_list(), df_full[i].to_list(), df_imputed[i].to_list()
        #print(min(list_f), max(list_f))
        list1, list2 = [list_f[j] for j in range(len(list_m)) if str(list_m[j]) == "nan"], [list_i[j] for j in range(len(list_m)) if str(list_m[j]) == "nan"]
        if i[0:3] == "cat":
            labels = list(set(list_f))
            print_confusionmatrix(list1, list2, labels, i, missing, cindex, mindex, model)
            
        else:
            plot_continuousscatter(list1, list2, i, missing, cindex, mindex, model)
            