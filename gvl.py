"""
Author: Alex
-----------------------------------------------------
Global (Light) vs Local Female using Logistic Regression, XGBoost, SVM, Linear SVM, and Random Forest
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

brier_scores = []
fairness_penalties = []
f_and_as = []
brier_scores_g = []
fairness_penalties_g = []
f_and_as_g = []


def main():
    ########## nan value approaches ##########
    # df = pd.read_pickle("Local_F_drop_nan_columns.pkl")
    # df = pd.read_pickle("Local_F_drop_nan_cases.pkl")
    # df = pd.read_pickle("Local_F_mean_imputation.pkl")
    # df = pd.read_pickle("Local_F_missForest.pkl")
    # df = pd.read_pickle("Local_F_mean_imputation_new_PUMA.pkl")
    df_local = pd.read_pickle("Local_F_mean_imputation_multi_cat.pkl")
    # df = pd.read_pickle("Local_F_mean_imputation_multi_PUMA_cat.pkl")
    ########## nan value approaches ##########

    # df_global = pd.read_pickle("Global_mean_imputation_multi_cat.pkl")
    df_global = pd.read_pickle("Global_Data_Mean_Impu.pkl")
    ran_state = [5566, 2266, 22, 66, 521, 1126, 36, 819, 23, 1225]

    for i in tqdm(range(len(ran_state))):
        ########## random under sampling ###########
        # rus = RandomUnderSampler(random_state=ran_state)
        # X_rus, y_rus = rus.fit_resample(X, y)
        # X_train, X_val, y_train, y_val = train_test_split(X_rus, y_rus, test_size=0.4, stratify=X_rus[:, 1],
        #                                                   random_state=ran_state)
        ########## random under sampling ###########

        ########### random over sampling ###########
        # ros = RandomOverSampler(random_state=ran_state)
        # X_ros, y_ros = ros.fit_resample(X, y)
        # X_train, X_val, y_train, y_val = train_test_split(X_ros, y_ros, test_size=0.4, stratify=X_ros[:, 1],
        #                                                   random_state=ran_state)
        ########### random over sampling ###########

        ########### random sampling ###########
        X_train, X_test, y_train, y_test = train_test_split(df_local.iloc[:, :-1], df_local['Recidivism_Arrest_Year1'],
                                                            test_size=0.4, stratify=df_local.iloc[:, 2],
                                                            random_state=ran_state[i])
        ########### random sampling ###########

        # split test in global
        for i in range(len(X_test)):
            df_global.drop(df_global[df_global['ID'] == X_test['ID'].iloc[i]].index, inplace=True)

        # slice ID away
        X_train_l = X_train.to_numpy()
        X_train_l = X_train_l[:, 1:]
        y_train_l = y_train.to_numpy()
        y_train_l = y_train_l.astype('int')

        data_g = df_global.to_numpy()
        X_train_g = data_g[:, 1: -1]
        y_train_g = data_g[:, -1]
        y_train_g = y_train_g.astype('int')
        ########## random oversampling ##########
        # ros = RandomOverSampler()
        # X_train_g, y_train_g = ros.fit_resample(X_train_g, y_train_g)
        ########## random oversampling ##########

        ########## oversampling on gender ##########
        X_train_g, y_train_g = over_sample(X_train_g, y_train_g)
        y_train_g = y_train_g.astype('int')
        ########## oversampling on gender ##########

        X_test = X_test.to_numpy()
        X_test = X_test[:, 1:]
        y_test = y_test.to_numpy()
        y_test = y_test.astype('int')

        local_female_model = local_female(X_train_l, y_train_l)
        global_model_ = global_model(X_train_g, y_train_g)
        print('\nlocal female metrics')
        metrics(local_female_model, X_test, y_test)
        print('\nglobal model metrics')
        metrics_g(global_model_, X_test, y_test)

    print('#' * 50 + ' local summary' + '#' * 50)
    print(f'mean brier: {stat.mean(brier_scores)}; se: {stat.stdev(brier_scores) / math.sqrt(len(ran_state))}')
    print(f'mean fairness Penalty: {stat.mean(fairness_penalties)}; se: {stat.stdev(fairness_penalties) / math.sqrt(len(ran_state))}')
    print(f'mean fairness and accurate: {stat.mean(f_and_as)}; se: {stat.stdev(f_and_as) / math.sqrt(len(ran_state))}')
    print('#' * 50 + ' global summary' + '#' * 50)
    print(f'mean brier: {stat.mean(brier_scores_g)}; se: {stat.stdev(brier_scores_g) / math.sqrt(len(ran_state))}')
    print(f'mean fairness Penalty: {stat.mean(fairness_penalties_g)}; se: {stat.stdev(fairness_penalties_g) / math.sqrt(len(ran_state))}')
    print(f'mean fairness and accurate: {stat.mean(f_and_as_g)}; se: {stat.stdev(f_and_as_g) / math.sqrt(len(ran_state))}')
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

    female_data_1 = np.concatenate((female_data, female_data, female_data, female_data, female_data, female_data), axis=0)
    female_data_2 = np.concatenate((female_data, female_data, female_data, female_data, female_data, female_data[:1181, :]), axis=0)
    female_data_over = np.concatenate((female_data_1, female_data_2), axis=0)
    np.random.shuffle(female_data_over)
    X_train = female_data_over[:, :-1]
    y_train = female_data_over[:, -1]
    return X_train, y_train


def local_female(X_train, y_train):
    """
    This function trains the local female model
    :param X_train:
    :param y_train:
    :return: model
    """
    ########### models ###########
    # model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=0.9, max_iter=10000)
    model = xgb.XGBClassifier()
    # model = SVC(probability=True)
    # model = LinearSVC(max_iter=1000000)
    # model = CalibratedClassifierCV(model)
    # model = RandomForestClassifier()
    ########### models ###########
    model.fit(X_train, y_train)
    # feature_importance(logit)
    return model


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
    # model = SVC(probability=True)
    # model = LinearSVC(max_iter=1000000)
    # model = CalibratedClassifierCV(model)
    # model = RandomForestClassifier()
    ########### models ###########
    model.fit(X_train, y_train)
    # feature_importance(logit)
    return model


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
