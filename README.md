# epilepsy_eeg_classification

epilepsy_eeg_classification is a python project that works with EEG data to classify epilepsy events. 

In this repo you will find resources to:
* Preprocess of raw EEG data
* Extrat time domain and frequency domain features
* Implementation of six popular ML classification models selected to tackle nonlinear and imbalanced data (MLP, KNN, Kernel SVM, Random Forest, AdaBoost, lasso-Logistic Regression)

## Paper 
This study is based on course project paper titled '
* Goal 1: Compare ML classification model's performance on single patient classifications
* Goal 2: Compare patient-specific and all-fit-one classification models for seizure classification

The complete study can be found in `report.pdf`

## Data Source
[CHB-MIT Scalp EEG Database] (https://physionet.org/content/chbmit/1.0.0/) [1]

## Scripts
`scripts/preprocessing.py` Inputs raw EEG files, performs high and low pass bandwidth filters, epoch segmentation, and feature extraction

`scripts/consolidation.py` Used to combine multiple preprocessed datasets of same or different subjects into a combined dataset 

`scripts/eeg_classifcation.py` Runs classification algorithms for a given dataset

`scripts/pyeeg.py` Useful functions to extra time-domain and frequency-domain features from raw EEG [2]

## Preprocessed data used in this study:

`data/chb01.csv` 145610 rows x 506 columns, 505 with seizure

`data/chb02.csv` 125685 row x 509 columns, 16 with seizure 

`data/chb03.csv` 136464 rows x 509 columns,

`data/five_Subjects.csv` 17955 rows x 509 cols, 430 with seizure

## References
[1] Ali Shoeb. Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD Thesis, Massachusetts Institute of Technology, September 2009.

[2] Forrest Sheng Bao, Xin Liu, Christina Zhang, "PyEEG: An Open Source Python Module for EEG/MEG Feature Extraction", Computational Intelligence and Neuroscience, vol. 2011, Article ID 406391, 7 pages, 2011. https://doi.org/10.1155/2011/406391

