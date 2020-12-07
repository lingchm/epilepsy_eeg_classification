######################################################################
# ISYE6740 Project
# EEG epilepsy classification
# classification algorithms
#
# @author Jintian Lu, Lingchao Mao
# @date 10/11/2020
######################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import time
from statistics import mean
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


### read in data
multiple_patients = True
ID = False
input_file = "data/five_subjects_35.csv"
output_file = "output/five_subjects_35_results.csv"

# prepare data
data = pd.read_csv(input_file)
X = data.loc[:, data.columns != "seizure"]
X = X.loc[:, X.columns != "start_time"]
X = X.loc[:, X.columns != "file ID"]
Y = np.asarray(data['seizure'])
feature_names = X.columns.tolist()
print('The number of samples for the non-seizure class is:', Y.shape[0])
print('The number of samples for the seizure class is:', np.sum(Y))

# if multiple patients, one-hot encode patient ID
if multiple_patients:
    X = X.loc[:, X.columns != "subject"] 
    if ID:
        patient = pd.get_dummies(data['subject'], prefix='subject')
        X = pd.concat([X, patient], axis = 1)


### preprocessing

# check zero variance features
thresholder = VarianceThreshold(threshold=0)
print("Variables Kept after removing features with 0 variance: ", thresholder.fit_transform(X).shape[1])

# highly correlated features
corr = abs(X.corr())
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
cols = [column for column in upper.columns if any(upper[column] < 0.9)]
print("Variables Kept after removing features with corr > 0.9: ", len(cols)) 

# normalize features
X = preprocessing.normalize(X)

# split into testing and training 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
print('The number of samples for the non-seizure class in training is:', y_train.shape[0])
print('The number of samples for the seizure class in training is:', np.sum(y_train))
print('The number of samples for the non-seizure class in testing is:', y_test.shape[0])
print('The number of samples for the seizure class in testing is:', np.sum(y_test))

'''
# feature selection
select_feature1 = SelectKBest(k=200, score_func = f_classif).fit(X_train, y_train)
X_train1 = select_feature1.transform(X_train)
X_test1 = select_feature1.transform(X_test)
print("Selected features (f-test): ", [feature_names[val] for val in select_feature1.get_support(indices=True).tolist()])
select_feature2 = SelectKBest(k=50, score_func = mutual_info_classif).fit(X_train, y_train)
X_train2 = select_feature2.transform(X_train)
X_test2 = select_feature2.transform(X_test)
print("Selected features (mutual info): ", [feature_names[val] for val in select_feature2.get_support(indices=True).tolist()])

# PCA
pca = PCA(n_components = 50)
X_train3 = pca.fit_transform(X_train)
X_test3 = pca.transform(X_test)
print("PCA explained variance is: ", np.sum(pca.explained_variance_ratio_))

# Oversampling
#from imblearn.over_sampling import SMOTE
#sm = SMOTE(ratio = 1.0)
#X_train_oversamp, y_train_oversamp = sm.fit_sample(X_train, y_train)
'''

### Modeling
names = ["MLP", "Kernel SVM", "Random Forest", "AdaBoost", "KNN"]

models = [
    MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(10, 10), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False),
    SVC(kernel="rbf", class_weight={1: 100}, random_state = 0),
    RandomForestClassifier(min_samples_split = 10, class_weight={1: 100}, random_state=0),
    AdaBoostClassifier(random_state=0),
    KNeighborsClassifier(2)
    ]

result = []
for name, model in zip(names, models):
    
    # cross validation
    kf = KFold(n_splits=5)
    accuracy, TPR, FPR = [], [], []
    for train, test in kf.split(X_train):
        model.fit(X_train[train, :], y_train[train])
        pred = model.predict(X_train[test])
        tn, fp, fn, tp = confusion_matrix(y_train[test], pred).ravel()
        accuracy.append((tp + tn)/(tn + fp + fn + tp))
        TPR.append(tp / (tp + fn))
        FPR.append(fp / (fp + tn))

    # hold our validation
    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    test_accuracy = ((tp + tn)/(tn + fp + fn + tp))
    test_TPR = (tp / (tp + fn))
    test_FPR = (fp / (fp + tn))
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print(name)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    print(confusion_matrix(y_test, pred))

    if name == "Adaboost":
        print(model.feature_importances_)

    # record results
    result.append ({
                'model': name,
                'cv accuracy': mean(accuracy),
                'cv TPR': mean(TPR),
                'cv FPR': mean(FPR),
                'test accuracy': test_accuracy,
                'test TPR': test_TPR,
                'test FPR': test_FPR
            })

final = pd.DataFrame.from_dict(result)
final.to_csv(os.path.join(output_file), index=False) 
print(final)
print("done")


