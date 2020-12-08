# epilepsy_eeg_classification

epilepsy_eeg_classification is a python project that works with EEG data to classify epilepsy events. 

In this repo you will find resources to:
* Preprocessing of raw EEG data
* Extration time domain and frequency domain features from EEG data
* Implementation of five popular ML classification models selected to tackle nonlinear and imbalanced data (MLP, KNN, Kernel SVM, Random Forest, AdaBoost)

## Paper 
This study is based on course project paper titled '
* Goal 1: Compare the performance of five popular nonlinear ML algorithms on patient-specific seizure classifications 
* Goal 2: Compare patient-specific versus non-patient specific classification performance

The complete study can be found in `report.pdf`

## Data Source
[CHB-MIT Scalp EEG Database] (https://physionet.org/content/chbmit/1.0.0/) [1]

## Scripts
`scripts/preprocessing.py` Inputs raw EEG files, performs high and low pass bandwidth filters, epoch segmentation, and feature extraction

`scripts/consolidation.py` Combines multiple preprocessed datasets of same or different subjects into a combined dataset 

`scripts/eeg_classifcation.py` Runs five seizure classification algorithms for a given dataset

`scripts/pyeeg.py` Useful functions to extra time-domain and frequency-domain features from raw EEG [2]

## Preprocessed data used in this study:
Raw EEG files from CHB-MIT database were preprocessed by `preprocessing.py` and saved into the `data` folder. Then, `consolidation.py` was used to combine 35-45 hour long files into the following datasets used for model training. These were not uploaded due to Github file size limits.

`data/chb01.csv` 145610 epochs, 505 with seizure

`data/chb02.csv` 125685 epochs, 207 with seizure 

`data/chb03.csv` 136464 epochs, 465 with seizure

`data/five_Subjects.csv` 125685 epochs, 430 with seizure

## References
[1] Ali Shoeb. Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment. PhD Thesis, Massachusetts Institute of Technology, September 2009.

[2] Forrest Sheng Bao, Xin Liu, Christina Zhang, "PyEEG: An Open Source Python Module for EEG/MEG Feature Extraction", Computational Intelligence and Neuroscience, vol. 2011, Article ID 406391, 7 pages, 2011. https://doi.org/10.1155/2011/406391

