# A comparison of Deep Learning and conventional statistical methods for missing data imputation

This is a thesis project about comparing imputation performances between deep learning methods and conventional statistical methods. In this project, GAIN and VAE with One-Hot and trainable embeddings for categorical variables were built for deep learning methods. MICE and Miss-Forest were chosen for representing conventional statistical methods.

Here is the link of our [paper](https://github.com/EagerSun/DL-vs-Stat_Impute/blob/main/paper/Deep%20Learning%20vs%20Conventional%20Stat%20in%20Missing%20Data%20Imputation.pdf).
## Requirements:

To run the code of this project normally, the packages below would be required:
### Python code:
python == 3.7

tensorflow == 1.15.0

tensorflow_probability == 0.8.0

numpy

pandas

matplotlib

sickit-learn

### R code:
R == 4.0.3

mice

missForest

MASS

Amelia

missMDA

softImpute

parallel

mltools

gdata

devtools

ggplot2

tidyr, dplyr

## Data Processing:
### The original dataset applied in research:
The full datasets picked or simulated for this project had been attached in `./data_stored/data` in this code file. To simplify the coding, the complex names of datasets were abondoned and rename with numbers increased from 1, which is also called case index here. The detail name of dataset for each indexed file is shown below:
1 -> DTI$cca
2 -> PimaIndiansDiabetes
3 -> spam
4 -> Carcinoma
5 -> DNA
6 -> NCbirths
7 -> VietNamI
8 -> sample1
9 -> sample2
10 -> sample3

It should be mentioned that dataset 8~10 are the simulated dataset sample from designed Gaussian distributions, the detail of construction of these three datasets were shown in `./Rcode/sample_data.R`.

### Add new datasets for methods comparison:
It is absolutely OK to add new full datasets(.csv) in `./data_stored/data` for comparing methods on the new datasets. The detail insert steps are very sample: Firstly, you need to rename the file with a postive integer as the case index. Secondly, the feature types of different features have to be marked out: Consider a column/feature name, if the certain feature is continuous feature, prefix `con_` would be required to added on the feature's name. Well, for categorical feature, prefix would be changed to `cat_`. Thirdly, add relative codes in `.\experiments.py` which have been well marked in the files. However, these prefixs could only have effect on deep learning models. For conventional methods, you could add relative codes or case index in `.\Rcode\Impute.R` and `run_r_test.py` which have been well marked in the file.

### Sampling missing datasets from full datasets:
In this research, sampleing missing datasets from full version could be finished by simply run `./Rcode/sample_miss.R`. You could also added new steps to sampling missing from new full datasets by adding code in correspond location which had been marked out in `./Rcode/sample_miss.R`.
Consider the different missing patterns and different missing ratio for the full data, the reult missing data files would be stored in address format as `./data_stored/data_miss/{missing pattern}/{case index}/{missing ratio}/{file index}.csv`.

## Data Imputation:
### Impute by deep learning models:
To start training the deep learning models for different dataset cases, just simply run `run_dl_test.py`. The result completed data for each missing data and summary of performance would be stored in different path. For completed data, it will stored in address format as `./data_stored/data_imputed/{missing pattern}/{case index}/{missing ratio}/{file index}.csv`. For summary of performance, it will stored in address format as `./performance/{missing pattern}/{case index}/{missing ratio}/{model name}_summary.csv`, In this file, "medCon" denotes the median of continuous feature measures and "qRCon" denotes the qR range of continuous feature measures, and "meanCon" denotes the mean of continuous feature measures and "varCon" denotes the standard variance of continuous feature measures. The remains is for categorical features.

### Impute by conventional statistical methods:
Initially, run the `./Rcode/Impute.R` to start imputation by conventional methods. After the code is finished, `mice_comnine.py` would be run to pooling the 100 multiple imputation results from MICE. Then, `run_r_test.py` would be run for generate summary files for different imputing cases. The summary files would also be stored in format as `./performance/{missing pattern}/{case index}/{missing ratio}/{model name}_summary.csv`.

## Code Running:
Linux server is highly recommended for running the code of this research. Initially, you need to use `cd` command is fix the workspace into the code address.

For python code, the code in file could be run in format in command line as `python \address\to\file.py &`.

For R code, the code could be run in command line as `R CMD BATCH \address\to\file.py &`.

## Parameters Tuning:
For deep learning methods, detail structure parameters, e.g. balanced parameters, noise ranges or layers nodes, could be fixed in `./parameters/Parameters_setting.py`. Training parameters, e.g. epochs/batch size, # of testing and sample testing could be fixed in `experiments.py`.

For conventional statistical methods, parameters tunning could be finished in `./Rcode/Imputed.R`. It should be mentioend that multi-processing was applied in imputations under conventional statistical methods, where several or lots of CPU could be required. If the computing resources are not enough, the `mc.cores` could be properly fixed.


## Resource
[GAIN resources from Jinsung Yoon](https://github.com/jsyoon0823/GAIN)

[VAE resources from JT McCoy](https://github.com/ProcessMonitoringStellenboschUniversity/IFAC-VAE-Imputation)

[Features Spliting resources from rcamino](https://github.com/rcamino/imputation-dgm)

