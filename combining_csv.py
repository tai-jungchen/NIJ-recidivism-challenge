"""
Author: Alex
-----------------------------------------------------
Combine male and female prediction
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
    df_male = pd.read_csv("Third_Year_Forcast/Third_Year_Male_Prediction.csv")
    df_female = pd.read_csv("Third_Year_Forcast/Third_Year_Female_Prediction.csv")
    ########## nan value approaches ##########

    male_data = df_male.to_numpy()
    female_data = df_female.to_numpy()

    output = np.concatenate((male_data, female_data), axis=0)
    output_df = pd.DataFrame(output, columns=['ID', 'Probability'])
    output_df.to_csv('Year_3_Forecast.csv', index=False)

    print('#' * 50 + ' end ' + '#' * 50)


if __name__ == '__main__':
    main()
