######################################################################
# ISYE6740 Project
# EEG epilepsy classification
# consolidates multiple dataset of one subject
#
# @author Jintian Lu, Lingchao Mao
# @date 11/14/2020
######################################################################
import numpy as np
import pandas as pd
import os

#### Consolidate multiple 1 hour files into one patient file 

'''
# edit file path here
folder = "data" 
#subjects = ["chb01", "chb02", "chb03"]
subjects = ["chb02"]

for subject in subjects:
    
    # locate subject folder
    subfolder = os.path.join(folder, subject)
    files = [file for file in os.listdir(subfolder) if file.endswith(".csv")]
    print(files)

    # concadenate files
    frames = []
    for filename in files:
        file = os.path.join(subfolder, filename)
        data = pd.read_csv(file)
        data['file ID'] = filename[6:8]
        frames.append(data)
    res = pd.concat(frames)
    res.to_csv(os.path.join(folder, subject + '.csv'), index=False) 

print("done")

'''

#### Multiple patients file
folder = "data" 
subjects = ["chb01\\chb01_01.csv", "chb01\\chb01_02.csv", "chb01\\chb01_03.csv", 
            "chb01\\chb01_05.csv", "chb01\\chb01_06.csv", "chb01\\chb01_07.csv", 
            "chb01\\chb01_08.csv", 
            "chb02\\chb02_01.csv", "chb02\\chb02_02.csv", "chb02\\chb02_03.csv", 
            "chb02\\chb02_04.csv", "chb02\\chb02_05.csv", "chb02\\chb02_06.csv", 
            "chb02\\chb02_19.csv", 
            "chb03\\chb03_01.csv", "chb03\\chb03_05.csv", "chb03\\chb03_06.csv", 
            "chb03\\chb03_07.csv", "chb03\\chb03_08.csv", "chb03\\chb03_09.csv", 
            "chb03\\chb03_10.csv", 
            "chb01\\chb01_46.csv", "chb01\\chb01_43.csv", "chb01\\chb01_42.csv", 
            "chb01\\chb01_34.csv", "chb01\\chb01_33.csv", "chb01\\chb01_32.csv", 
            "chb05\\chb05_06.csv", 
            "chb02\\chb02_26.csv", "chb02\\chb02_27.csv", "chb03\\chb03_33.csv", 
            "chb03\\chb03_30.csv", "chb03\\chb03_31.csv", "chb03\\chb03_32.csv", 
            "chb08\\chb08_02.csv"]

all = []
for subject in subjects:
    
    # locate subject folder
    file = os.path.join(folder, subject)

    # concadenate files
    data = pd.read_csv(file)
    data['subject'] = subject
    all.append(data)

res = pd.concat(all)
res.to_csv(os.path.join(folder, 'five_subjects_35.csv'), index=False) 

print("done")



    


