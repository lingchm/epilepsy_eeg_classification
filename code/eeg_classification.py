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
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, RFECV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA, LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import f_regression, mutual_info_regression

## read in data
files = ["chb01_03"]
data = pd.read_csv('data/preprocessed/chb01_03.csv')
X = data.loc[:, data.columns != "seizure"]
X = X.loc[:, X.columns != "start_time"]
Y = data['seizure']
feature_names = X.columns.tolist()

# visualize class distribution
ax = sn.countplot(Y,label="Count")
non_seizure, seizure = Y.value_counts()
print('The number of trials for the non-seizure class is:', non_seizure)
print('The number of trials for the seizure class is:', seizure)
# plt.show()

# normalization
X = preprocessing.scale(X)

'''
# PCA
pca = PCA(n_components = 10)
X = pca.fit_transform(X)
print("PCA explained variance is: ", pca.explained_variance_ratio_)
'''

# split into testing and training
# TO DO need to make sure equal proportion??
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
non_seizure, seizure = y_train.value_counts()
print('The number of trials for the non-seizure class in training is:', non_seizure)
print('The number of trials for the seizure class in training is:', seizure)
non_seizure, seizure = y_test.value_counts()
print('The number of trials for the non-seizure class in testing is:', non_seizure)
print('The number of trials for the seizure class in testing is:', seizure)

# feature selection
select_feature = SelectKBest(k=40).fit(X_train, y_train)
print("Selected features: ", [feature_names[val] for val in select_feature.get_support(indices=True).tolist()])
mi = mutual_info_regression(X, Y)
f_test, _ = f_regression(X, Y)

# Oversampling????
#from imblearn.over_sampling import SMOTE
#sm = SMOTE(ratio = 1.0)
#X_train_oversamp, y_train_oversamp = sm.fit_sample(X_train, y_train)

##### Modeling
# neural networks
mlp = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(10, 10), learning_rate='constant',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=None,
       shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
       verbose=False, warm_start=False)
mlp.fit(X_train, y_train)
score = mlp.score(X_test, y_test)
y_pred = mlp.predict(X_test)
print("Neural Network MLP", "score : ", score)
print(confusion_matrix(y_test, y_pred))


# other models
names = ["Nearest Neighbors", "Linear SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "AdaBoost", "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(2),
    SVC(kernel="linear", C=0.025),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    XGBClassifier()]

classifier_score=[]
for name, classifier in zip(names, classifiers):
    classifier.fit(X_train, y_train)
    score = classifier.score(X_test, y_test)
    print(classifier, "; score : ", score)
    classifier_score.append([score,name])
    y_pred = classifier.predict(X_test)
    # accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10,)
    # print("mean accuracy : ", accuracies.mean())
    print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred, target_names=['Non-seizure', 'Seizure']))

# LDA
lda = LDA(n_components = 2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

# Fitting Logistic Regression to the Training set
classifier = LogisticRegression()
classifier.fit(X_train_lda, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test_lda)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=['Non-seizure', 'Seizure']))


