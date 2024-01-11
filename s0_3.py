"""
Author: Alex
-----------------------------------------------------
Global model using over sampling on gender Logistic Regression, XGBoost, SVM, Linear SVM, and Random Forest
Using the year 1 + year 2 + year 3 as labels
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
accs = []


def main():
    df = pd.read_pickle("Processed_Data/stage3_df.pkl")
    ran_states = [5566, 2266, 22, 66, 521, 1126, 36, 819, 23, 1225]

    for ran_state in ran_states:
        # predict year123
        data = df.to_numpy()
        X_pre = data[:, 1: -6]
        y12 = data[:, -1]
        y3 = data[:, -3]
        X = np.concatenate((X_pre, y12[:, np.newaxis], y3[:, np.newaxis]), axis=1)
        y_123 = data[:, -6]
        y_123 = y_123.astype('int')

        X_train, X_val, y_train, y_val = train_test_split(X, y_123, test_size=0.4, stratify=X[:, 0], random_state=ran_state)
        y_train = y_train.astype('int')

        Xy_val = np.concatenate((X_val, y_val[:, np.newaxis]), axis=1)
        Xy_val = Xy_val[Xy_val[:, 0] == -1]     # select female cases
        # Xy_val = Xy_val[Xy_val[:, -3] != 1]     # drop year12 positive cases in test
        X_val = Xy_val[:, : -3]
        y3_val = Xy_val[:, -2]
        y3_val = y3_val.astype('int')

        global_model_ = global_model(X_train[:, :-2], y_train)
        y_123_pred = global_model_.predict(X_val)

        # cancel out year1
        y_3_pred = y_123_pred - Xy_val[:, -3]
        acc = np.sum(y_3_pred == y3_val) / len(y_3_pred)
        accs.append(acc)
        print(f'*****Accuracy of year123 model: {acc}')

    print('#' * 50 + ' global summary' + '#' * 50)
    print(f'mean accuracy of year 123 method: {stat.mean(accs)}; se: {stat.stdev(accs) / math.sqrt(10)}')
    print('#' * 50 + ' end ' + '#' * 50)


def brier_score_improve(y_pred, y_true):
    """
    This function search the best y_pred for brier score
    :return: best brier score
    """
    pps = np.linspace(0.0, 1.0, num=11)     # candidate prob for prediction 1
    pns = np.linspace(0.0, 1.0, num=11)     # candidate prob for prediction 0
    opt_prob = np.zeros(len(y_pred))
    brier_scores_indices = []
    brier_scores = []

    # search best param #
    for i in range(len(pps)):
        for j in range(len(pns)):
            opt_prob = np.where(y_pred == 0, pns[j], pps[i])
            # opt_prob = np.where(y_pred == 1, pns[j], pps[i])
            brier_score = brier_score_loss(y_true, opt_prob)
            brier_scores_indices.append((brier_score, i, j))
            brier_scores.append(brier_score)
    best_brier_param = min(brier_scores_indices, key=lambda t: t[0])
    print(f'best brier score: {best_brier_param[0]}')
    print(f'best probability for prediction of 1: {pps[best_brier_param[1]]}')
    print(f'best probability for prediction of 0: {pns[best_brier_param[2]]}')
    # plot #
    plt.plot(np.linspace(1, len(pps)*len(pns), num=len(pps)*len(pns)), brier_scores)
    plt.ylabel('brier_score')
    plt.xlabel('parameter sets')
    plt.title('brier score to optimized probabily plot')
    # plt.show()
    print()


def global_model(X_train, y_train):
    """
    This function trains the global model
    :param X_train:
    :param y_train:
    :return: model
    """
    ########### models ###########
    model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.4, max_iter=1000)
    # model = xgb.XGBClassifier()
    # model = RandomForestClassifier()
    # model = SVC(probability=True)
    # model = LinearSVC(max_iter=1000000)
    # model = CalibratedClassifierCV(model)
    ########### models ###########
    model.fit(X_train, y_train)
    # feature_importance(logit)
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
    # brier_score_improve(y_pred, y_test)
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
