"""
Author: Alex
-----------------------------------------------------
Global model (scenario 7)
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
    ########## dataset ##########
    df_global = pd.read_pickle("Processed_Data/s5.pkl")
    df_test = pd.read_pickle("Processed_Data/s5_testing.pkl")
    # df_global = pd.read_pickle("Global_mean_imputation_multi_cat.pkl")
    # df_global = pd.read_pickle("Local_F_mean_imputation.pkl")
    ########## dataset ##########

    data = df_global.to_numpy()
    Xy = data[:, 1:]
    X_train = data[:, 1:-3]
    y = data[:,  -1]
    y = y.astype('int')

    X_test = df_test.to_numpy()

    global_model_ = global_model(X_train, y)
    y_pred = global_model_.predict(X_test[:, 1:])
    y_prob = global_model_.predict_proba(X_test[:, 1:])
    X_test_idp = np.concatenate((X_test, y_pred[:, np.newaxis], y_prob), axis=1)
    y_prob_new, y_pred_new, test_id = ensemble_35(X_test_idp, Xy)

    test_id = np.expand_dims(test_id, axis=1)
    y_prob_out = np.expand_dims(y_prob_new[:, 1], axis=1)
    output = np.concatenate((test_id, y_prob_out), axis=1)
    output_df = pd.DataFrame(output, columns=['ID', 'Probability'])
    # output_df.to_csv('Second_Year_Female_Prediction.csv', index=False)

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


def ensemble_35(Xy_val_y1y2_p, Xy_train_y1y2):
    """
    This function implement the ensemble model of scenario 3 and 5
    :param Xy: data set with predicted label
    :return: new y_pred and new y_prob
    """
    Xy_val_3 = Xy_val_y1y2_p[Xy_val_y1y2_p[:, -4] == 2]        # select the false positive cases
    Xy_val_5 = Xy_val_y1y2_p[Xy_val_y1y2_p[:, -4] != 2]        # select the FF and FT
    test_id_3 = Xy_val_3[:, 0]
    test_id_5 = Xy_val_5[:, 0]
    ########## s3 model ##########
    Xy_train_y1y2_3= Xy_train_y1y2[Xy_train_y1y2[:, -3] != 1]       # drop Recidivism_Arrest_Year1 positive cases
    X_train_3 = Xy_train_y1y2_3[:, :-3]       # drop Recidivism_Arrest_Year1 column
    y_train_3 = Xy_train_y1y2_3[:, -2]
    model_3 = xgb.XGBClassifier()
    model_3.fit(X_train_3, y_train_3)
    feature_importance(model_3)
    y_pred_3 = model_3.predict(Xy_val_3[:, 1:-4])
    y_prob_3 = model_3.predict_proba(Xy_val_3[:, 1:-4])
    ########## s3 model ##########

    ########## s5 model ##########
    Xy_train_y1y2_5 = Xy_train_y1y2
    X_train_5 = Xy_train_y1y2_5[:, :-3]
    y_train_5 = Xy_train_y1y2_5[:, -1]
    model_5 = xgb.XGBClassifier()
    model_5.fit(X_train_5, y_train_5)
    y_pred_5 = model_5.predict(Xy_val_5[:, 1:-4])
    y_prob_5 = model_5.predict_proba(Xy_val_5[:, 1:-4])
    y_prob_5_bin = y_prob_adj(y_prob_5)
    ########## s5 model ##########

    ########## combine results ##########
    y_prob = np.concatenate((y_prob_3, y_prob_5_bin), axis=0)
    y_pred = np.concatenate((y_pred_3, y_pred_5), axis=0)
    test_id = np.concatenate((test_id_3, test_id_5), axis=0)
    ########## combine results ##########

    return y_prob, y_pred, test_id


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


if __name__ == '__main__':
    main()
