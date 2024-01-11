"""
Author: Alex
-----------------------------------------------------
Local female models
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from missingpy import MissForest
from tqdm import tqdm
import matplotlib.pyplot as plt
import statistics as stat
import math
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, brier_score_loss

brier_scores = []
fairness_penalties = []
f_and_as = []


def main():
    ########## nan value approaches ##########
    # df = pd.read_pickle("Local_F_drop_nan_columns.pkl")
    # df = pd.read_pickle("Local_F_drop_nan_cases.pkl")
    # df = pd.read_pickle("Local_F_mean_imputation.pkl")
    # df = pd.read_pickle("Local_F_missForest.pkl")
    # df = pd.read_pickle("Local_F_mean_imputation_new_PUMA.pkl")
    # df = pd.read_pickle("Local_F_mean_imputation_multi_cat.pkl")
    # df = pd.read_pickle("Local_F_mean_imputation_multi_PUMA_cat.pkl")
    df = pd.read_pickle("Processed_Data/Global_MI_multi_cat_2.pkl")
    ########## nan value approaches ##########

    df = df[df['Gender'] == -1]
    data = df.to_numpy()
    X = data[:, 1:-1]
    y = data[:, -1]
    y = y.astype('int')

    ########## RF imputation ##########
    # imputer = MissForest()
    # X = imputer.fit_transform(X)
    ########## RF imputation ##########

    ran_state = [5566, 2266, 22, 66, 521, 1126, 36, 819, 23, 1225]

    for i in tqdm(range(len(ran_state))):
        ########## random under sampling ###########
        # rus = RandomUnderSampler(random_state=ran_state[i])
        # X_rus, y_rus = rus.fit_resample(X, y)
        # X_train, X_val, y_train, y_val = train_test_split(X_rus, y_rus, test_size=0.4, stratify=X_rus[:, 1],
        #                                                   random_state=ran_state[i])
        ########## random under sampling ###########

        ########### random over sampling ###########
        # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, stratify=X[:, 1], random_state=ran_state[i])
        # ros = RandomOverSampler(random_state=ran_state[i])
        # X_train, y_train = ros.fit_resample(X_train, y_train)
        ########### random over sampling ###########

        ########### random sampling ###########
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, stratify=X[:, 1],
                                                          random_state=ran_state[i])
        ########### random sampling ###########

        ########### models ###########
        model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.4, max_iter=1000)
        # model = xgb.XGBClassifier()
        # model = SVC(probability=True)
        # model = LinearSVC(max_iter=1e7)
        # model = CalibratedClassifierCV(model)
        # model = RandomForestClassifier()
        ########### models ###########

        model.fit(X_train, y_train)
        metrics(model, X_val, y_val)
        # feature_importance(model)

    print('#' * 50 + ' summary' + '#' * 50)
    print(f'mean brier: {stat.mean(brier_scores)}; se: {stat.stdev(brier_scores)/math.sqrt(len(ran_state))}')
    print(f'mean fairness Penalty: {stat.mean(fairness_penalties)}; se: {stat.stdev(fairness_penalties) / math.sqrt(len(ran_state))}')
    print(f'mean fairness and accurate: {stat.mean(f_and_as)}; se: {stat.stdev(f_and_as) / math.sqrt(len(ran_state))}')

    print('#'*50 + ' end ' + '#'*50)


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
    importance = model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    plt.title('Feature Importance Plot')
    plt.xlabel('Features')
    plt.ylabel('Feature importance')
    plt.show()


def metrics(model, X_test, y_test):
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
    brier_scores.append(bs)
    fairness_penalties.append(fairness_penalty)
    f_and_as.append(f_and_a)


if __name__ == '__main__':
    main()
