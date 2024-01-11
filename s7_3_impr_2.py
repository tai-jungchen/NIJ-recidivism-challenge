"""
Author: Alex
-----------------------------------------------------
Global model training specifically on false positive cases - include TFF and FTF in training data
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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from sklearn.model_selection import KFold

brier_scores_g = []
fairness_penalties_g = []
f_and_as_g = []

brier_scores_c = []
fairness_penalties_c = []
f_and_as_c = []


def main():
    ########## dataset ##########
    # df_global = pd.read_pickle("Processed_Data/stage3_df.pkl")
    df_global = pd.read_pickle("Processed_Data/stage3_s8_df.pkl")
    ########## dataset ##########

    data = df_global.to_numpy()
    X_pre = data[:, 1: -7]
    X_new = data[:, -1]
    X = np.concatenate((X_pre, X_new[:, np.newaxis]), axis=1)
    y3 = data[:, -4]
    Xy3 = np.concatenate((X, y3[:, np.newaxis]), axis=1)
    y = data[:,  -3]
    y = y.astype('int')

    ran_state = [5566, 2266, 22, 66, 521, 1126, 36, 819, 23, 1225]

    for i in tqdm(range(len(ran_state))):
        ########## sampling ##########
        X_train_y3, X_val_y3, y_train, y_val = train_test_split(Xy3, y, test_size=0.4, stratify=X[:, 0],
                                                          random_state=ran_state[i])
        X_train = X_train_y3[:, :-1]     # drop year_3 column
        y_train = y_train.astype('int')
        Xy_train_y3 = np.concatenate((X_train_y3, y_train[:, np.newaxis]), axis=1)

        # get only female and year 1 year 2 recidivism != 1 #
        Xy_val_y3 = np.concatenate((X_val_y3, y_val[:, np.newaxis]), axis=1)
        # Xy_val_y3 = Xy_val_y3[Xy_val_y3[:, 0] == -1]     # discard male #
        Xy_val_y3 = Xy_val_y3[Xy_val_y3[:, -1] != 2]     # discard previous positive subject
        X_val = Xy_val_y3[:, :-2]
        ########## sampling ##########

        global_model_ = global_model(X_train, y_train)
        y_pred = global_model_.predict(X_val)
        y_prob = global_model_.predict_proba(X_val)
        Xy_val_y3_p = np.concatenate((Xy_val_y3, y_pred[:, np.newaxis], y_prob), axis=1)
        y_prob_new, y_pred_new, y_true_new, X_test = ensemble_35(Xy_val_y3_p, Xy_train_y3)
        print('\nglobal model metrics')
        metrics_g(y_prob_new, y_pred_new, y_true_new, X_test)

    print('#' * 50 + ' global summary' + '#' * 50)
    print(f'mean brier: {stat.mean(brier_scores_g)}; se: {stat.stdev(brier_scores_g) / math.sqrt(len(ran_state))}')
    print(f'mean fairness Penalty: {stat.mean(fairness_penalties_g)}; se: {stat.stdev(fairness_penalties_g) / math.sqrt(len(ran_state))}')
    print(f'mean fairness and accurate: {stat.mean(f_and_as_g)}; se: {stat.stdev(f_and_as_g) / math.sqrt(len(ran_state))}')

    print('#' * 50 + ' year 12 false positive summary' + '#' * 50)
    print(f'mean brier: {stat.mean(brier_scores_c)}; se: {stat.stdev(brier_scores_c) / math.sqrt(len(ran_state))}')
    print(
        f'mean fairness Penalty: {stat.mean(fairness_penalties_c)}; se: {stat.stdev(fairness_penalties_c) / math.sqrt(len(ran_state))}')
    print(
        f'mean fairness and accurate: {stat.mean(f_and_as_c)}; se: {stat.stdev(f_and_as_c) / math.sqrt(len(ran_state))}')
    print('#' * 50 + ' end ' + '#' * 50)


def global_model(X_train, y_train):
    """
    This function trains the global model
    :param X_train:
    :param y_train:
    :return: model
    """
    y_train = y_train.astype('int')
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


def metrics_g(y_prob, y_pred, y_true, X_test):
    """
    This function calculates the metrics
    :param model:
    """
    y_pred = y_pred.astype('int')

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    fnr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    fpr = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    fpe = cm[1, 0] / (cm[0, 0] + cm[1, 0])
    spe = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    ope = (cm[0, 1] + cm[1, 0]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    bs = brier_score_loss(y_true, y_prob[:, 1])
    fairness_penalty = fairness_penalty_helper(y_pred, y_true, X_test[:, 1])
    f_and_a = (1 - bs) * fairness_penalty
    print('#' * 50)
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


def over_sample(X, y):
    """
    This function do over sampling based on gender
    :param X: X_train
    :param y: y_train
    :return: oversampled (X_train, y_train)
    """
    Xy = np.concatenate((X, y[:, np.newaxis]), axis=1)
    pos_data = Xy[Xy[:, -1] == 1]
    neg_data = Xy[Xy[:, -1] == 0]

    ########## P/N ratio adjustment ##########
    # P/N = 0.3304
    # pos_data_1 = pos_data
    # pos_data_2 = pos_data[:441, :]

    # P/N = 0.5
    # pos_data_1 = np.concatenate((pos_data, pos_data), axis=0)
    # pos_data_2 = np.concatenate((pos_data_1, pos_data[:149, :]), axis=0)

    # P/N = 0.4
    pos_data_1 = pos_data
    pos_data_2 = pos_data[:759, :]
    ########## P/N ratio adjustment ##########

    pos_data_over = np.concatenate((pos_data_1, pos_data_2), axis=0)
    over_sam = np.concatenate((pos_data_over, neg_data), axis=0)
    np.random.shuffle(over_sam)
    X_train = over_sam[:, :-1]
    y_train = over_sam[:, -1]
    return X_train, y_train


def train_fp_cases(Xy):
    """
    This function use 2-fold cross validation to pick up the false positive cases for training
    :param X: Original X_train
    :param y: Original y_train
    :return: X_train, y_train for false positive cases only
    """
    X = Xy[:, :-1]
    y = Xy[:, -1]

    ########## 2-fold cross val ##########
    # X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X, y, test_size=0.5, stratify=X[:, 0], random_state=521)
    # model_1 = global_model(X_train_1[:, :-1], y_train_1)
    # y_pred_2 = model_1.predict(X_train_2[:, :-1])
    # fp_cases_2 = X_train_2[np.logical_and(y_pred_2 == 2, y_train_2 != 2)]
    # model_2 = global_model(X_train_2[:, :-1], y_train_2)
    # y_pred_1 = model_2.predict(X_train_1[:, :-1])
    # fp_cases_1 = X_train_1[np.logical_and(y_pred_1 == 2, y_train_1 != 2)]
    # fp_cases = np.concatenate((fp_cases_1, fp_cases_2), axis=0)
    # return fp_cases[:, :-1], fp_cases[:, -1]
    ########## 2-fold cross val ##########

    ########## 5-fold cross val ##########
    kf = KFold(n_splits=5)
    fp_cases = []
    for train_index, test_index in kf.split(X):
        model = global_model(X[train_index, :-1], y[train_index])
        y_pred = model.predict(X[test_index, :-1])
        X_test = X[test_index, :]
        fp_case = X_test[np.logical_and(y_pred == 2, y[test_index] != 2)]
        fp_cases.append(fp_case)
    # concatenate fp cases #
    fp_cases_last = np.concatenate((fp_cases[0], fp_cases[1], fp_cases[2], fp_cases[3], fp_cases[4]), axis=0)
    return fp_cases_last[:, :-1], fp_cases_last[:, -1]
    ########## 5-fold cross val ##########


def ensemble_35(Xy_val_y3_p, Xy_train_y3):
    """
    This function implement the ensemble model of scenario 3 and 5
    :param Xy_val_y3_p: data set with predicted label and probability
    :param Xy_train_y3: training data set
    :return: new y_pred and new y_prob
    """
    # Splitting false positive and non false positive #
    Xy_val_f = Xy_val_y3_p[Xy_val_y3_p[:, -4] == 2]        # select the false positive cases (Group A)
    Xy_val_t = Xy_val_y3_p[Xy_val_y3_p[:, -4] != 2]        # select the FFT and FFF (Group B and C)
    Xy_val_35 = np.concatenate((Xy_val_f, Xy_val_t), axis=0)

    ########### training set for scenario 3 ###########
    # training on complete data set (exclude year 12 positive cases) #
    Xy_train_y3_3 = Xy_train_y3[Xy_train_y3[:, -1] != 2]  # drop Recidivism_Arrest_Year1 2 positive cases #
    X_train_3 = Xy_train_y3_3[:, :-2]
    y_train_3 = Xy_train_y3_3[:, -2]
    y_train_3 = y_train_3.astype('int')

    # training only on false positive cases #
    # X_train_3, y_train_3 = train_fp_cases(Xy_train_y3)
    # y_train_3 = y_train_3.astype('int')
    ########### training set for scenario 3 ###########

    ########## oversampling / undersampling ##########
    # X_train_3, y_train_3 = over_sample(X_train_3, y_train_3)
    # oversample = RandomOverSampler(random_state=521)
    # X_train_3, y_train_3 = oversample.fit_resample(X_train_3, y_train_3)
    # undersample = RandomUnderSampler(random_state=521)
    # X_train_3, y_train_3 = undersample.fit_resample(X_train_3, y_train_3)
    ########## oversampling / undersampling ##########

    ########## s3 model on false positive cases ##########
    model_3_f = global_model(X_train_3, y_train_3)
    y_pred_3_f = model_3_f.predict(Xy_val_f[:, :-6])
    y_prob_3_f = model_3_f.predict_proba(Xy_val_f[:, :-6])
    y_val_3_f = Xy_val_f[:, -6].astype('int')
    ########## s3 model on false positive cases ##########

    ########## s3 model on non false positive cases ##########
    model_3_t = global_model(X_train_3, y_train_3)
    y_pred_3_t = model_3_t.predict(Xy_val_t[:, :-6])
    y_prob_3_t = model_3_t.predict_proba(Xy_val_t[:, :-6])
    y_val_3_t = Xy_val_t[:, -6].astype('int')
    ########## s3 model on non false positive cases ##########

    # training set for scenario 5 #
    Xy_train_y3_5 = Xy_train_y3
    X_train_5 = Xy_train_y3_5[:, :-2]
    y_train_5 = Xy_train_y3_5[:, -1]
    y_train_5 = y_train_5.astype('int')

    ########## oversampling / undersampling ##########
    # oversample = RandomOverSampler(random_state=521)
    # X_train_5, y_train_5 = oversample.fit_resample(X_train_5, y_train_5)
    # undersample = RandomUnderSampler(random_state=521)
    # X_train_5, y_train_5 = undersample.fit_resample(X_train_5, y_train_5)
    ########## oversampling / undersampling ##########

    ########## s5 model on non false positive cases ##########
    model_5_t = global_model(X_train_5, y_train_5)
    y_pred_5_t = model_5_t.predict(Xy_val_t[:, :-6])
    y_prob_5_t = model_5_t.predict_proba(Xy_val_t[:, :-6])
    y_prob_5_bin_t = y_prob_adj(y_prob_5_t)
    y_val_5_t = Xy_val_t[:, -5].astype('int')
    ########## s5 model on non false positive cases ##########

    ########## s5 model on false positive cases ##########
    model_5_f = global_model(X_train_5, y_train_5)
    y_pred_5_f = model_5_f.predict(Xy_val_f[:, :-6])
    y_prob_5_f = model_5_f.predict_proba(Xy_val_f[:, :-6])
    y_prob_5_bin_f = y_prob_adj(y_prob_5_f)
    y_val_5_f = Xy_val_f[:, -5].astype('int')
    ########## s5 model on false positive cases ##########

    diging(y_prob_3_f, y_val_3_f, y_pred_3_f, y_prob_3_t, y_val_3_t, y_pred_3_t,
           y_prob_5_bin_f, y_val_5_f, y_pred_5_f, y_prob_5_bin_t, y_val_5_t, y_pred_5_t, Xy_val_f, Xy_val_t)

    ########## combine results ##########
    y_prob = np.concatenate((y_prob_3_f, y_prob_5_bin_t), axis=0)
    y_pred = np.concatenate((y_pred_3_f, y_pred_5_t), axis=0)
    y_true = np.concatenate((y_val_3_f, y_val_5_t), axis=0)
    ########## combine results ##########

    return y_prob, y_pred, y_true, Xy_val_35


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


def y_prob_adj(y_prob):
    """
    This function rescale the probability of label 1 and label 0
    :param y_prob: (nparray) list (699*3) of probability
    :return: adjusted y_prob (699*2)
    """
    y_prob_bin = np.zeros((y_prob.shape[0], 2))
    y_prob_bin[:, 0] = y_prob[:, 0] / (y_prob[:, 0] + y_prob[:, 1])
    y_prob_bin[:, 1] = 1 - y_prob_bin[:, 0]
    return y_prob_bin


def diging(y_prob_3_f, y_val_3_f, y_pred_3_f, y_prob_3_t, y_val_3_t, y_pred_3_t,
           y_prob_5_bin_f, y_val_5_f, y_pred_5_f, y_prob_5_bin_t, y_val_5_t, y_pred_5_t, Xy_val_f, Xy_val_t):
    """
    This function do some digging
    :param y_prob: y_prob after rescaling to two classes
    :param y_true: True label of the subject
    :return:
    """
    delta_prob = y_prob_3_f[:, 0] - y_prob_3_f[:, 1]
    min_gaps_indices = sorted(range(len(delta_prob)), key=lambda i: delta_prob[i])[:8]
    y_true = y_val_3_f[min_gaps_indices]
    y_prob = y_prob_3_f[min_gaps_indices]

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(y_prob_3_f[:, 0])
    axs[0, 0].plot(y_prob_3_f[:, 1])
    axs[0, 0].plot(y_val_3_f, 'o')
    axs[0, 0].set_title('false positive group on scenario 3')
    axs[0, 0].set(ylabel='Probability')

    axs[0, 1].plot(y_prob_3_t[:, 0])
    axs[0, 1].plot(y_prob_3_t[:, 1])
    axs[0, 1].plot(y_val_3_t, 'o')
    axs[0, 1].set_title('non false positive group on scenario 3')
    axs[0, 1].set(ylabel='Probability')

    axs[1, 0].plot(y_prob_5_bin_f[:, 0])
    axs[1, 0].plot(y_prob_5_bin_f[:, 1])
    axs[1, 0].plot(y_val_5_f, 'o')
    axs[1, 0].set_title('false positive group on scenario 5')
    axs[1, 0].set(xlabel='cases', ylabel='Probability')

    axs[1, 1].plot(y_prob_5_bin_t[:, 0])
    axs[1, 1].plot(y_prob_5_bin_t[:, 1])
    axs[1, 1].plot(y_val_5_t, 'o')
    axs[1, 1].set_title('non false positive group on scenario 5')
    axs[1, 1].set(xlabel='cases', ylabel='Probability')

    # ax1.legend(['probability of not recidivating in year 3', 'probability of recidivating in year 3',
    #             'True answer of whether the subject recidivated in year 3'])
    # plt.show()

    print(f'********** metrics for false positive on scenario 3 **********')
    metrics_c(y_prob_3_f, y_val_3_f, y_pred_3_f, Xy_val_f)
    # print(f'\n********** metrics for non false positive on scenario 3 **********')
    # metrics_c(y_prob_3_t, y_val_3_t, y_pred_3_t, Xy_val_t)
    # print(f'\n********** metrics for false positive on scenario 5 **********')
    # y_pred_5_f = np.where(y_prob_5_bin_f[:, 1] > 0.5, 1, 0)
    # metrics_c(y_prob_5_bin_f, y_val_5_f, y_pred_5_f, Xy_val_f)
    # print(f'\n********** metrics for non false positive on scenario 5 **********')
    # metrics_c(y_prob_5_bin_t, y_val_5_t, y_pred_5_t, Xy_val_t)


def metrics_c(y_prob, y_test, y_pred, X_test):
    """
    This function calculates the metrics
    :param model:
    """
    y_pred = y_pred.astype('int')
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    fnr = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    fpr = cm[1, 0] / (cm[1, 0] + cm[1, 1])
    fpe = cm[1, 0] / (cm[0, 0] + cm[1, 0])
    spe = cm[0, 1] / (cm[0, 1] + cm[1, 1])
    ope = (cm[0, 1] + cm[1, 0]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    bs = brier_score_loss(y_test, y_prob[:, 1])
    fairness_penalty = fairness_penalty_helper(y_pred, y_test, X_test[:, 1])
    f_and_a = (1 - bs) * fairness_penalty
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


if __name__ == '__main__':
    main()
