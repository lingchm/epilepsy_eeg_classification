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

# file path here
folder = "data"
subjects = ["chb01"]

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



    


