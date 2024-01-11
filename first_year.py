"""
Author: Alex
-----------------------------------------------------
First year Prediction on female via Global model + over sampling on gender + Logistic Regression
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
    # df = pd.read_csv('First_Year_Forecast/VT-ISE_1YearForecast.csv')
    df_global = pd.read_pickle("Processed_Data/Global_mean_imputation_multi_cat.pkl")
    df_test = pd.read_pickle('Processed_Data/first_year_testing.pkl')
    ########## nan value approaches ##########

    data = df_global.to_numpy()
    X_train = data[:, 1:data.shape[1] - 1]
    y_train = data[:, data.shape[1] - 1]
    y_train = y_train.astype('int')

    test_data = df_test.to_numpy()
    X_test_id = test_data[:, 0]
    X_test = test_data[:, 1:]

    ########## oversampling on gender ##########
    X_train, y_train = over_sample(X_train, y_train)
    y_train = y_train.astype('int')
    ########## oversampling on gender ##########

    global_model_ = global_model(X_train, y_train)
    y_prob = global_model_.predict_proba(X_test)
    y_pred = global_model_.predict(X_test)

    X_test_id = np.expand_dims(X_test_id, axis=1)
    y_prob_out = np.expand_dims(y_prob[:, 1], axis=1)
    output = np.concatenate((X_test_id, y_prob_out), axis=1)
    output_df = pd.DataFrame(output, columns=['ID', 'Probability'])
    # output_df.to_csv('First_Year_Female_Prediction.csv', index=False)

    print('#' * 50 + ' end ' + '#' * 50)


def over_sample(X, y):
    """
    This function do over sampling based on gender
    :param X: X_train
    :param y: y_train
    :return: oversampled (X_train, y_train)
    """
    y = np.expand_dims(y, axis=1)
    X = np.concatenate((X, y), axis=1)
    female_data = X[X[:, 0] == -1]
    male_data = X[X[:, 0] == 1]

    female_data_1 = np.concatenate((female_data, female_data, female_data, female_data, female_data, female_data), axis=0)
    female_data_2 = np.concatenate((female_data, female_data[:292, :]), axis=0)
    female_data_over = np.concatenate((female_data_1, female_data_2), axis=0)
    over_sam = np.concatenate((female_data_over, male_data), axis=0)

    np.random.shuffle(over_sam)
    X_train = over_sam[:, :-1]
    y_train = over_sam[:, -1]
    return X_train, y_train


def global_model(X_train, y_train):
    """
    This function trains the global model
    :param X_train:
    :param y_train:
    :return: model
    """
    ########### models ###########
    model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.4, max_iter=1000)
    ########### models ###########
    model.fit(X_train, y_train)
    # feature_importance(logit)
    return model


def feature_importance(model):
    """
    This function plots the feature importance of the model
    :param model:
    """
    importance = model.coef_[0]
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.title('Feature Importance Plot')
    plt.xlabel('feature')
    plt.ylabel('feature importance')
    plt.show()


if __name__ == '__main__':
    main()
