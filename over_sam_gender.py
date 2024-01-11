"""
Author: Alex
-----------------------------------------------------
Global model predicting second year recidivism (scenario 1)
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
    df_global = pd.read_pickle("Processed_Data/Global_MI_multi_cat_year12_no1_2.pkl")
    # df_global = pd.read_pickle("Global_mean_imputation_multi_cat.pkl")
    # df_global = pd.read_pickle("Local_F_mean_imputation.pkl")
    ########## dataset ##########

    data = df_global.to_numpy()
    X = data[:, 1:- 3]
    y = data[:,  - 2]
    y = y.astype('int')

    ran_state = [5566, 2266, 22, 66, 521, 1126, 36, 819, 23, 1225]

    for i in tqdm(range(len(ran_state))):
        ########## oversampling on gender ##########
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, stratify=X[:, 0],
                                                          random_state=ran_state[i])
        # X_train, y_train = over_sample(X_train, y_train)
        y_train = y_train.astype('int')

        y_val = np.expand_dims(y_val, axis=1)
        Xy_val = np.concatenate((X_val, y_val), axis=1)
        Xy_val = Xy_val[Xy_val[:, 0] == -1]
        X_val = Xy_val[:, :- 1]
        y_val = Xy_val[:, - 1]
        y_val = y_val.astype('int')
        ########## oversampling on gender ##########

        global_model_ = global_model(X_train, y_train)
        print('\nglobal model metrics')
        metrics_g(global_model_, X_val, y_val)

    print('#' * 50 + ' global summary' + '#' * 50)
    print(f'mean brier: {stat.mean(brier_scores_g)}; se: {stat.stdev(brier_scores_g) / math.sqrt(len(ran_state))}')
    print(f'mean fairness Penalty: {stat.mean(fairness_penalties_g)}; se: {stat.stdev(fairness_penalties_g) / math.sqrt(len(ran_state))}')
    print(f'mean fairness and accurate: {stat.mean(f_and_as_g)}; se: {stat.stdev(f_and_as_g) / math.sqrt(len(ran_state))}')
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
    female_data_2 = np.concatenate((female_data, female_data[:176, :]), axis=0)
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
