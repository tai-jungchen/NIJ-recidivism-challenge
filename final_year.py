"""
Author: Alex
-----------------------------------------------------
Global model (scenario 8) - based on scenario 4 final version
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

brier_scores_g = []
fairness_penalties_g = []
f_and_as_g = []


def main():
    ########## dataset ##########
    df_global = pd.read_pickle("Processed_Data/stage3_s8_df.pkl")
    df_test = pd.read_pickle("Processed_Data/third_year_testing.pkl")
    ########## dataset ##########

    data = df_global.to_numpy()
    X_pre = data[:, 1: -7]
    three_cat_2 = data[:, -1]
    X_train = np.concatenate((X_pre, three_cat_2[:, np.newaxis]), axis=1)
    y_train = data[:,  -4]
    y_train = y_train.astype('int')

    X_test = df_test.to_numpy()
    X_test_id = X_test[:, 0]

    global_model_ = global_model(X_train, y_train)
    y_pred = global_model_.predict(X_test[:, 1:])
    y_prob = global_model_.predict_proba(X_test[:, 1:])

    X_test_id = np.expand_dims(X_test_id, axis=1)
    y_prob_out = np.expand_dims(y_prob[:, 1], axis=1)
    output = np.concatenate((X_test_id, y_prob_out), axis=1)
    output_df = pd.DataFrame(output, columns=['ID', 'Probability'])
    output_df.to_csv('Third_Year_Female_Prediction.csv', index=False)

    print('#' * 50 + ' end ' + '#' * 50)


def global_model(X_train, y_train):
    """
    This function trains the global model
    :param X_train:
    :param y_train:
    :return: model
    """
    ########### models ###########
    # model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.4, max_iter=1000)
    model = xgb.XGBClassifier()
    # model = RandomForestClassifier()
    # model = SVC(probability=True)
    # model = LinearSVC(max_iter=1000000)
    # model = CalibratedClassifierCV(model)
    ########### models ###########
    model.fit(X_train, y_train)
    # feature_importance(model)
    return model


def metrics_g(model, X_test, y_test):
    """
    This function calculates the metrics
    :param model:
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    fnr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    fpr = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    fpe = cm[1, 0] / (cm[0, 0] + cm[1, 0])
    spe = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    ope = (cm[0, 1] + cm[1, 0]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    bs = brier_score_loss(y_test, y_prob[:, 1])
    fairness_penalty = fairness_penalty_helper(y_pred, y_test, X_test[:, 1])
    f_and_a = (1 - bs) * fairness_penalty
    print('#' * 10 + str(model) + '#' * 10)
    print(f'Brier Score: {bs}')
    print(f'fairness penalty: {fairness_penalty}')
    print(f'fair and accurate: {f_and_a}')
    print(f"\nThe confusion matrix:\n {cm}")
    print(f'False Negative Rate: {fnr}')
    print(f'False Positive Rate: {fpr}')
    print(f'Failure Prediction Error: {fpe}')
    print(f'Success Prediction Error: {spe}')
    print(f'Overall Prediction Error: {ope}')
    brier_scores_g.append(bs)
    fairness_penalties_g.append(fairness_penalty)
    f_and_as_g.append(f_and_a)


def fairness_penalty_helper(y_pred, y_true, race):
    """
    This function help calculate the False Positive Rate according to the Race
    :param y_pred: (nparray) predicted labels
    :param y_true: (nparray) true labels
    :param race: (nparray) race of the example
    :return:
    """
    fp_black = np.where((y_pred == 1) & (y_true == 0) & (race == 1))
    fp_black = len(fp_black[0])
    tn_black = np.where((y_pred == 0) & (y_true == 0) & (race == 1))
    tn_black = len(tn_black[0])
    fpr_black = fp_black / (fp_black + tn_black)

    fp_white = np.where((y_pred == 1) & (y_true == 0) & (race == -1))
    fp_white = len(fp_white[0])
    tn_white = np.where((y_pred == 0) & (y_true == 0) & (race == -1))
    tn_white = len(tn_white[0])
    fpr_white = fp_white / (fp_white + tn_white)
    fairness_penalty = 1 - abs(fpr_black - fpr_white)
    return fairness_penalty


def feature_importance(model):
    """
    This function plots the feature importance of the model
    :param model:
    """
    # importance = model.coef_[0]
    importance = model.feature_importances_
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
