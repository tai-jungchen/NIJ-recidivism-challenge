"""
Author: Alex
-----------------------------------------------------
Global model (scenario 8) - based on scenario 4
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

brier_scores_c = []
fairness_penalties_c = []
f_and_as_c = []


def main():
    ########## dataset ##########
    df_global = pd.read_pickle("Processed_Data/stage3_s8_df.pkl")
    ########## dataset ##########

    data = df_global.to_numpy()
    X_pre = data[:, 1: -7]
    three_cat_2 = data[:, -1]
    X = np.concatenate((X_pre, three_cat_2[:, np.newaxis]), axis=1)
    y = data[:,  -4]
    y = y.astype('int')

    ran_state = [5566, 2266, 22, 66, 521, 1126, 36, 819, 23, 1225]

    for i in tqdm(range(len(ran_state))):
        ########## sampling ##########
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, stratify=X[:, 0], random_state=ran_state[i])
        y_train = y_train.astype('int')

        # get only female and year 1 year 2 recidivism != 1 #
        Xy_val = np.concatenate((X_val, y_val[:, np.newaxis]), axis=1)
        # Xy_val = Xy_val[Xy_val[:, 0] == -1]     # discard male
        Xy_val = Xy_val[Xy_val[:, -2] == 0]     # discard year 1 year 2 positive subjects
        X_val = Xy_val[:, :-1]
        y_val = Xy_val[:, -1]
        y_val = y_val.astype('int')
        ########## sampling ##########

        global_model_ = global_model(X_train, y_train)
        print('\nglobal model metrics')
        metrics_g(global_model_, X_val, y_val)

    print('#' * 50 + ' global summary' + '#' * 50)
    print(f'mean brier: {stat.mean(brier_scores_g)}; se: {stat.stdev(brier_scores_g) / math.sqrt(len(ran_state))}')
    print(f'mean fairness Penalty: {stat.mean(fairness_penalties_g)}; se: {stat.stdev(fairness_penalties_g) / math.sqrt(len(ran_state))}')
    print(f'mean fairness and accurate: {stat.mean(f_and_as_g)}; se: {stat.stdev(f_and_as_g) / math.sqrt(len(ran_state))}')

    # print('#' * 50 + ' year 1 false positive summary' + '#' * 50)
    # print(f'mean brier: {stat.mean(brier_scores_c)}; se: {stat.stdev(brier_scores_c) / math.sqrt(len(ran_state))}')
    # print(
    #     f'mean fairness Penalty: {stat.mean(fairness_penalties_c)}; se: {stat.stdev(fairness_penalties_c) / math.sqrt(len(ran_state))}')
    # print(
    #     f'mean fairness and accurate: {stat.mean(f_and_as_c)}; se: {stat.stdev(f_and_as_c) / math.sqrt(len(ran_state))}')
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


def diging(y_prob, y_true, y_pred, X_test):
    """
    This function do some digging
    :param y_prob: y_prob after rescaling to two classes
    :param y_true: True label of the subject
    :return:
    """
    plt.plot(y_prob[:, 0])
    plt.plot(y_prob[:, 1])
    plt.plot(y_true, 'o')
    plt.grid()
    plt.title('probability of prediction')
    plt.xlabel('cases')
    plt.ylabel('probability')
    plt.legend(['probability of not recidivating in year 3', 'probability of recidivating in year 3',
                'True answer of whether the subject recidivated in year 3'])
    plt.show()
    # metrics_c(y_prob_fp, y_true_fp, y_pred_fp, X_test)


def metrics_c(y_prob, y_test, y_pred, X_test):
    """
    This function calculates the metrics
    :param model:
    """
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    fnr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    fpr = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    fpe = cm[1, 0] / (cm[0, 0] + cm[1, 0])
    spe = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    ope = (cm[0, 1] + cm[1, 0]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    bs = brier_score_loss(y_test, y_prob[:, 1])
    fairness_penalty = fairness_penalty_helper(y_pred, y_test, X_test[:, 1])
    f_and_a = (1 - bs) * fairness_penalty
    print('#' * 10 + 'Year 1 false positive cases' + '#' * 10)
    print(f'Brier Score: {bs}')
    print(f'fairness penalty: {fairness_penalty}')
    print(f'fair and accurate: {f_and_a}')
    print(f"\nThe confusion matrix:\n {cm}")
    print(f'False Negative Rate: {fnr}')
    print(f'False Positive Rate: {fpr}')
    print(f'Failure Prediction Error: {fpe}')
    print(f'Success Prediction Error: {spe}')
    print(f'Overall Prediction Error: {ope}')
    brier_scores_c.append(bs)
    fairness_penalties_c.append(fairness_penalty)
    f_and_as_c.append(f_and_a)
    print('#' * 10 + 'Year 1 false positive cases' + '#' * 10)


if __name__ == '__main__':
    main()
