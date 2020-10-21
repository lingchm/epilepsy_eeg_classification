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

## read in data
files = ["chb01_03"]
data = pd.read_csv('data/chb01_03.csv')
X = data.loc[:, data.columns != "seizure"]
Y = data['seizure']

# visualize class distribution
ax = sn.countplot(Y,label="Count")
non_seizure, seizure = Y.value_counts()
print('The number of trials for the non-seizure class is:', non_seizure)
print('The number of trials for the seizure class is:', seizure)
plt.show()

# normalization
X = preprocessing.scale(X)

# PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X)
print("PCA explained variance is: ", pca.explained_variance_ratio_)

# split into testing and training
# TO DO need to make sure equal proportion??
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# feature selection
select_feature = SelectKBest(k=10).fit(X_train, y_train)

# Oversampling????
from imblearn.over_sampling import SMOTE
sm = SMOTE(ratio = 1.0)
X_train_oversamp, y_train_oversamp = sm.fit_sample(X_train, y_train)


##### Modeling
names = ["Nearest Neighbors", "Linear SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes"]

classifiers = [
    KNeighborsClassifier(2),
    SVC(kernel="linear", C=0.025),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    XGBClassifier()]

clf_score=[]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        clf_score.append([score,name])
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10,)
        print(clf, " mean accuracy is: ", accuracies.mean())
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, target_names=['Non-seizure', 'Seizure']))

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


# Neural Networks
early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=10,verbose=1, mode='auto')

# Initialising the ANN
clf_ann = Sequential()

# Adding the input layer and the first hidden layer
clf_ann.add(Dense(activation="relu", kernel_initializer="uniform", units=100, input_dim=178))
clf_ann.add(BatchNormalization())
clf_ann.add(Dropout(0.5))

# Adding the second hidden layer
clf_ann.add(Dense(activation="relu", kernel_initializer="uniform", units=100))
clf_ann.add(BatchNormalization())
clf_ann.add(Dropout(0.5))

# Adding the output layer
clf_ann.add(Dense(units=1, kernel_initializer="uniform",  activation="sigmoid"))

# Compiling the ANN
clf_ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# loss = 'categorical_crossentropy'
# Fitting the ANN to the Training set
clf_ann.fit(X_train, y_train, batch_size = 32,epochs = 100, validation_data=(X_test,y_test), callbacks=[early_stop])

# Predicting the Test set results
y_pred = clf_ann.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=['Non-seizure', 'Seizure']))

