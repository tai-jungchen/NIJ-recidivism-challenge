"""
Author: Alex
-----------------------------------------------------
eda on testing
"""

import pandas as pd
import statistics as stat
import math
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, brier_score_loss


def main():
    ########## nan value approaches ##########
    df_year1 = pd.read_csv("Data/NIJ_s_Recidivism_Challenge_Test_Dataset1.csv")
    df_year2 = pd.read_csv("Data/NIJ_s_Recidivism_Challenge_Test_Dataset2.csv")
    df_year3 = pd.read_csv("Data/NIJ_s_Recidivism_Challenge_Test_Dataset3.csv")
    train_df = pd.read_csv("Data/NIJ_s_Recidivism_Challenge_Training_Dataset.csv")
    ########## nan value approaches ##########

    # 51. Recidivism_Arrest_Year1
    train_df.loc[(train_df.Recidivism_Arrest_Year1 == False), 'Recidivism_Arrest_Year1'] = 0
    train_df.loc[(train_df.Recidivism_Arrest_Year1 == True), 'Recidivism_Arrest_Year1'] = 1
    # 52. Recidivism_Arrest_Year2
    train_df.loc[(train_df.Recidivism_Arrest_Year2 == False), 'Recidivism_Arrest_Year2'] = 0
    train_df.loc[(train_df.Recidivism_Arrest_Year2 == True), 'Recidivism_Arrest_Year2'] = 1
    # 52. Recidivism_Arrest_Year2
    train_df.loc[(train_df.Recidivism_Arrest_Year3 == False), 'Recidivism_Arrest_Year3'] = 0
    train_df.loc[(train_df.Recidivism_Arrest_Year3 == True), 'Recidivism_Arrest_Year3'] = 1

    Recidivism_Arrest_Year123 = train_df['Recidivism_Arrest_Year1'] + train_df['Recidivism_Arrest_Year2'] + train_df['Recidivism_Arrest_Year3']
    train_df['Recidivism_Arrest_Year123'] = Recidivism_Arrest_Year123

    y1_id = df_year1['ID'].to_numpy()
    y2_id = df_year2['ID'].to_numpy()
    y3_id = df_year3['ID'].to_numpy()

    re12 = 0
    for i in range(len(y1_id)):
        for j in range(len(y2_id)):
            if y1_id[i] == y2_id[j]:
                re12 += 1
    print(f'remaining: {re12}')

    re23 = 0
    for i in range(len(y2_id)):
        for j in range(len(y3_id)):
            if y2_id[i] == y3_id[j]:
                re23 += 1
    print(f'remaining: {re23}')

    re13 = 0
    for i in range(len(y1_id)):
        for j in range(len(y3_id)):
            if y1_id[i] == y3_id[j]:
                re13 += 1
    print(f'remaining: {re13}')


    print('#' * 50 + ' end ' + '#' * 50)


if __name__ == '__main__':
    main()
