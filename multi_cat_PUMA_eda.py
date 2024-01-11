"""
Author: Alex
-----------------------------------------------------
original training set + multi-cat + Breakdown PUMA feature
"""

import pandas as pd
import numpy as np
from tqdm import tqdm


def main():
    puma = pd.read_pickle('for_PUMA_cat.pkl')
    df = pd.read_csv('Data/NIJ_s_Recidivism_Challenge_Training_Dataset_Multi-category features.csv')

    # drop useless columns #
    train_df1 = df.iloc[:, 53:57]
    puma = puma.drop(['Prior_Arrest_Episodes_DVCharges', 'Prior_Arrest_Episodes_GunCharges',
                              'Prior_Conviction_Episodes_Viol', '_v2', '_v3', '_v4', 'Prior_Revocations_Parole',
                              'Prior_Revocations_Probation', 'Condition_MH_SA', 'Condition_Cog_Ed',
                              'Condition_Other'], axis=1)
    train_df = pd.concat([puma, train_df1], axis=1)

    # swap label to the end
    titles = list(train_df.columns)
    for i in range(4):
        titles[38+i], titles[38+i+1] = titles[38+i+1], titles[38+i]
    train_df = train_df[titles]

    # encode content #
    # 2. Gender
    train_df.loc[(df.Gender == 'M'), 'Gender'] = 1
    train_df.loc[(df.Gender == 'F'), 'Gender'] = -1
    # 3. Race
    train_df.loc[(df.Race == 'BLACK'), 'Race'] = 1
    train_df.loc[(df.Race == 'WHITE'), 'Race'] = -1
    # 4. Age_at_Release
    train_df.loc[(df.Age_at_Release == '18-22'), 'Age_at_Release'] = 18
    train_df.loc[(df.Age_at_Release == '23-27'), 'Age_at_Release'] = 23
    train_df.loc[(df.Age_at_Release == '28-32'), 'Age_at_Release'] = 28
    train_df.loc[(df.Age_at_Release == '33-37'), 'Age_at_Release'] = 33
    train_df.loc[(df.Age_at_Release == '38-42'), 'Age_at_Release'] = 38
    train_df.loc[(df.Age_at_Release == '43-47'), 'Age_at_Release'] = 43
    train_df.loc[(df.Age_at_Release == '48 or older'), 'Age_at_Release'] = 48
    # 6. Gang_Affiliated
    train_df.loc[(df.Gang_Affiliated == True), 'Gang_Affiliated'] = 1
    train_df.loc[(df.Gang_Affiliated == False), 'Gang_Affiliated'] = 0
    # 8. Supervision_Level_First
    train_df.loc[(df.Supervision_Level_First == 'Standard'), 'Supervision_Level_First'] = 1
    train_df.loc[(df.Supervision_Level_First == 'Specialized'), 'Supervision_Level_First'] = 2
    train_df.loc[(df.Supervision_Level_First == 'High'), 'Supervision_Level_First'] = 3
    # 9. Education_Level
    train_df.loc[(df.Education_Level == 'At least some college'), 'Education_Level'] = 1
    train_df.loc[(df.Education_Level == 'High School Diploma'), 'Education_Level'] = 2
    train_df.loc[(df.Education_Level == 'Less than HS diploma'), 'Education_Level'] = 3
    # 10. Dependents
    train_df.loc[(df.Dependents == '0'), 'Dependents'] = 0
    train_df.loc[(df.Dependents == '1'), 'Dependents'] = 1
    train_df.loc[(df.Dependents == '2'), 'Dependents'] = 2
    train_df.loc[(df.Dependents == '3 or more'), 'Dependents'] = 3
    # 11. Prison_Offense
    train_df.loc[(df.Prison_Offense == 'Violent/Sex'), 'Prison_Offense'] = 5
    train_df.loc[(df.Prison_Offense == 'Violent/Non-Sex'), 'Prison_Offense'] = 4
    train_df.loc[(df.Prison_Offense == 'Property'), 'Prison_Offense'] = 3
    train_df.loc[(df.Prison_Offense == 'Drug'), 'Prison_Offense'] = 2
    train_df.loc[(df.Prison_Offense == 'Other'), 'Prison_Offense'] = 1
    # 12. Prison_Years
    train_df.loc[(df.Prison_Years == 'Less than 1 year'), 'Prison_Years'] = 1
    train_df.loc[(df.Prison_Years == '1-2 years'), 'Prison_Years'] = 2
    train_df.loc[(df.Prison_Years == 'Greater than 2 to 3 years'), 'Prison_Years'] = 3
    train_df.loc[(df.Prison_Years == 'More than 3 years'), 'Prison_Years'] = 4
    # 13. Prior_Arrest_Episodes_Felony
    train_df.loc[(df.Prior_Arrest_Episodes_Felony == '0'), 'Prior_Arrest_Episodes_Felony'] = 0
    train_df.loc[(df.Prior_Arrest_Episodes_Felony == '1'), 'Prior_Arrest_Episodes_Felony'] = 1
    train_df.loc[(df.Prior_Arrest_Episodes_Felony == '2'), 'Prior_Arrest_Episodes_Felony'] = 2
    train_df.loc[(df.Prior_Arrest_Episodes_Felony == '3'), 'Prior_Arrest_Episodes_Felony'] = 3
    train_df.loc[(df.Prior_Arrest_Episodes_Felony == '4'), 'Prior_Arrest_Episodes_Felony'] = 4
    train_df.loc[(df.Prior_Arrest_Episodes_Felony == '5'), 'Prior_Arrest_Episodes_Felony'] = 5
    train_df.loc[(df.Prior_Arrest_Episodes_Felony == '6'), 'Prior_Arrest_Episodes_Felony'] = 6
    train_df.loc[(df.Prior_Arrest_Episodes_Felony == '7'), 'Prior_Arrest_Episodes_Felony'] = 7
    train_df.loc[(df.Prior_Arrest_Episodes_Felony == '8'), 'Prior_Arrest_Episodes_Felony'] = 8
    train_df.loc[(df.Prior_Arrest_Episodes_Felony == '9'), 'Prior_Arrest_Episodes_Felony'] = 9
    train_df.loc[(df.Prior_Arrest_Episodes_Felony == '10 or more'), 'Prior_Arrest_Episodes_Felony'] = 10
    # 14. Prior_Arrest_Episodes_Misd
    train_df.loc[(df.Prior_Arrest_Episodes_Misd == '0'), 'Prior_Arrest_Episodes_Misd'] = 0
    train_df.loc[(df.Prior_Arrest_Episodes_Misd == '1'), 'Prior_Arrest_Episodes_Misd'] = 1
    train_df.loc[(df.Prior_Arrest_Episodes_Misd == '2'), 'Prior_Arrest_Episodes_Misd'] = 2
    train_df.loc[(df.Prior_Arrest_Episodes_Misd == '3'), 'Prior_Arrest_Episodes_Misd'] = 3
    train_df.loc[(df.Prior_Arrest_Episodes_Misd == '4'), 'Prior_Arrest_Episodes_Misd'] = 4
    train_df.loc[(df.Prior_Arrest_Episodes_Misd == '5'), 'Prior_Arrest_Episodes_Misd'] = 5
    train_df.loc[(df.Prior_Arrest_Episodes_Misd == '6 or more'), 'Prior_Arrest_Episodes_Misd'] = 6
    # 15. Prior_Arrest_Episodes_Violent
    train_df.loc[(df.Prior_Arrest_Episodes_Violent == '0'), 'Prior_Arrest_Episodes_Violent'] = 0
    train_df.loc[(df.Prior_Arrest_Episodes_Violent == '1'), 'Prior_Arrest_Episodes_Violent'] = 1
    train_df.loc[(df.Prior_Arrest_Episodes_Violent == '2'), 'Prior_Arrest_Episodes_Violent'] = 2
    train_df.loc[(df.Prior_Arrest_Episodes_Violent == '3 or more'), 'Prior_Arrest_Episodes_Violent'] = 3
    # 16. Prior_Arrest_Episodes_Property
    train_df.loc[(df.Prior_Arrest_Episodes_Property == '0'), 'Prior_Arrest_Episodes_Property'] = 0
    train_df.loc[(df.Prior_Arrest_Episodes_Property == '1'), 'Prior_Arrest_Episodes_Property'] = 1
    train_df.loc[(df.Prior_Arrest_Episodes_Property == '2'), 'Prior_Arrest_Episodes_Property'] = 2
    train_df.loc[(df.Prior_Arrest_Episodes_Property == '3'), 'Prior_Arrest_Episodes_Property'] = 3
    train_df.loc[(df.Prior_Arrest_Episodes_Property == '4'), 'Prior_Arrest_Episodes_Property'] = 4
    train_df.loc[(df.Prior_Arrest_Episodes_Property == '5 or more'), 'Prior_Arrest_Episodes_Property'] = 5
    # 17. Prior_Arrest_Episodes_Drug
    train_df.loc[(df.Prior_Arrest_Episodes_Drug == '0'), 'Prior_Arrest_Episodes_Drug'] = 0
    train_df.loc[(df.Prior_Arrest_Episodes_Drug == '1'), 'Prior_Arrest_Episodes_Drug'] = 1
    train_df.loc[(df.Prior_Arrest_Episodes_Drug == '2'), 'Prior_Arrest_Episodes_Drug'] = 2
    train_df.loc[(df.Prior_Arrest_Episodes_Drug == '3'), 'Prior_Arrest_Episodes_Drug'] = 3
    train_df.loc[(df.Prior_Arrest_Episodes_Drug == '4'), 'Prior_Arrest_Episodes_Drug'] = 4
    train_df.loc[(df.Prior_Arrest_Episodes_Drug == '5 or more'), 'Prior_Arrest_Episodes_Drug'] = 5
    # 18. _v1
    train_df.loc[(df._v1 == '0'), '_v1'] = 0
    train_df.loc[(df._v1 == '1'), '_v1'] = 1
    train_df.loc[(df._v1 == '2'), '_v1'] = 2
    train_df.loc[(df._v1 == '3'), '_v1'] = 3
    train_df.loc[(df._v1 == '4'), '_v1'] = 4
    train_df.loc[(df._v1 == '5 or more'), '_v1'] = 5
    # 21. Prior_Conviction_Episodes_Felony
    train_df.loc[(df.Prior_Conviction_Episodes_Felony == '0'), 'Prior_Conviction_Episodes_Felony'] = 0
    train_df.loc[(df.Prior_Conviction_Episodes_Felony == '1'), 'Prior_Conviction_Episodes_Felony'] = 1
    train_df.loc[(df.Prior_Conviction_Episodes_Felony == '2'), 'Prior_Conviction_Episodes_Felony'] = 2
    train_df.loc[(df.Prior_Conviction_Episodes_Felony == '3 or more'), 'Prior_Conviction_Episodes_Felony'] = 3
    # 22. Prior_Conviction_Episodes_Misd
    train_df.loc[(df.Prior_Conviction_Episodes_Misd == '0'), 'Prior_Conviction_Episodes_Misd'] = 0
    train_df.loc[(df.Prior_Conviction_Episodes_Misd == '1'), 'Prior_Conviction_Episodes_Misd'] = 1
    train_df.loc[(df.Prior_Conviction_Episodes_Misd == '2'), 'Prior_Conviction_Episodes_Misd'] = 2
    train_df.loc[(df.Prior_Conviction_Episodes_Misd == '3'), 'Prior_Conviction_Episodes_Misd'] = 3
    train_df.loc[(df.Prior_Conviction_Episodes_Misd == '4 or more'), 'Prior_Conviction_Episodes_Misd'] = 5
    # 24. Prior_Conviction_Episodes_Prop
    train_df.loc[(df.Prior_Conviction_Episodes_Prop == '0'), 'Prior_Conviction_Episodes_Prop'] = 0
    train_df.loc[(df.Prior_Conviction_Episodes_Prop == '1'), 'Prior_Conviction_Episodes_Prop'] = 1
    train_df.loc[(df.Prior_Conviction_Episodes_Prop == '2'), 'Prior_Conviction_Episodes_Prop'] = 2
    train_df.loc[(df.Prior_Conviction_Episodes_Prop == '3 or more'), 'Prior_Conviction_Episodes_Prop'] = 3
    # 25. Prior_Conviction_Episodes_Drug
    train_df.loc[(df.Prior_Conviction_Episodes_Drug == '0'), 'Prior_Conviction_Episodes_Drug'] = 0
    train_df.loc[(df.Prior_Conviction_Episodes_Drug == '1'), 'Prior_Conviction_Episodes_Drug'] = 1
    train_df.loc[(df.Prior_Conviction_Episodes_Drug == '2 or more'), 'Prior_Conviction_Episodes_Drug'] = 2
    # 34. Recidivism_Arrest_Year1
    train_df.loc[(df.Recidivism_Arrest_Year1 == False), 'Recidivism_Arrest_Year1'] = 0
    train_df.loc[(df.Recidivism_Arrest_Year1 == True), 'Recidivism_Arrest_Year1'] = 1
    # Set1_Prior_Arrest
    train_df.loc[(df.Set1_Prior_Arrest == 'FF'), 'Set1_Prior_Arrest'] = 1
    train_df.loc[(df.Set1_Prior_Arrest == 'FT'), 'Set1_Prior_Arrest'] = 2
    train_df.loc[(df.Set1_Prior_Arrest == 'TF'), 'Set1_Prior_Arrest'] = 3
    train_df.loc[(df.Set1_Prior_Arrest == 'TT'), 'Set1_Prior_Arrest'] = 4
    # Set2_Prior_Conviction
    train_df.loc[(df.Set2_Prior_Conviction == 'FFFF'), 'Set2_Prior_Conviction'] = 1
    train_df.loc[(df.Set2_Prior_Conviction == 'FFFT'), 'Set2_Prior_Conviction'] = 2
    train_df.loc[(df.Set2_Prior_Conviction == 'FFTF'), 'Set2_Prior_Conviction'] = 3
    train_df.loc[(df.Set2_Prior_Conviction == 'FTFF'), 'Set2_Prior_Conviction'] = 4
    train_df.loc[(df.Set2_Prior_Conviction == 'TFFF'), 'Set2_Prior_Conviction'] = 5
    train_df.loc[(df.Set2_Prior_Conviction == 'FFTT'), 'Set2_Prior_Conviction'] = 6
    train_df.loc[(df.Set2_Prior_Conviction == 'FTFT'), 'Set2_Prior_Conviction'] = 7
    train_df.loc[(df.Set2_Prior_Conviction == 'FTTF'), 'Set2_Prior_Conviction'] = 8
    train_df.loc[(df.Set2_Prior_Conviction == 'TTFF'), 'Set2_Prior_Conviction'] = 9
    train_df.loc[(df.Set2_Prior_Conviction == 'TFFT'), 'Set2_Prior_Conviction'] = 10
    train_df.loc[(df.Set2_Prior_Conviction == 'TFTF'), 'Set2_Prior_Conviction'] = 11
    train_df.loc[(df.Set2_Prior_Conviction == 'FTTT'), 'Set2_Prior_Conviction'] = 12
    train_df.loc[(df.Set2_Prior_Conviction == 'TFTT'), 'Set2_Prior_Conviction'] = 13
    train_df.loc[(df.Set2_Prior_Conviction == 'TTFT'), 'Set2_Prior_Conviction'] = 14
    train_df.loc[(df.Set2_Prior_Conviction == 'TTTF'), 'Set2_Prior_Conviction'] = 15
    train_df.loc[(df.Set2_Prior_Conviction == 'TTTT'), 'Set2_Prior_Conviction'] = 16
    # Set3_Prior_Revocations
    train_df.loc[(df.Set3_Prior_Revocations == 'FF'), 'Set3_Prior_Revocations'] = 1
    train_df.loc[(df.Set3_Prior_Revocations == 'FT'), 'Set3_Prior_Revocations'] = 2
    train_df.loc[(df.Set3_Prior_Revocations == 'TF'), 'Set3_Prior_Revocations'] = 3
    train_df.loc[(df.Set3_Prior_Revocations == 'TT'), 'Set3_Prior_Revocations'] = 4
    # Set4_Condition
    train_df.loc[(df.Set4_Condition == 'FFF'), 'Set4_Condition'] = 1
    train_df.loc[(df.Set4_Condition == 'FFT'), 'Set4_Condition'] = 2
    train_df.loc[(df.Set4_Condition == 'FTF'), 'Set4_Condition'] = 3
    train_df.loc[(df.Set4_Condition == 'TFF'), 'Set4_Condition'] = 4
    train_df.loc[(df.Set4_Condition == 'FTT'), 'Set4_Condition'] = 5
    train_df.loc[(df.Set4_Condition == 'TFT'), 'Set4_Condition'] = 6
    train_df.loc[(df.Set4_Condition == 'TTF'), 'Set4_Condition'] = 7
    train_df.loc[(df.Set4_Condition == 'TTT'), 'Set4_Condition'] = 8

    print(f'nan values:\n{train_df.isnull().sum()}')
    ########## drop nan columns ##########
    # train = train_df.drop(['Gang_Affiliated', 'Supervision_Risk_Score_First', 'Supervision_Level_First', 'Prison_Offense']
    #                       , axis=1)
    # train = train[train['Gender'] == -1]
    # train.to_pickle('Local_F_drop_nan_columns.pkl')
    ########## drop nan columns ##########

    ########## drop nan cases ##########
    # train_df = train_df.drop(['Gang_Affiliated'], axis=1)
    # train = train_df.dropna()
    # train = train[train['Gender'] == -1]
    # train.to_pickle('Local_F_drop_nan_cases.pkl')
    ########## drop nan cases ##########

    ########## mean imputation ###########
    train_df = train_df.drop(['Gang_Affiliated'], axis=1)
    column_means = train_df.mean()
    train = train_df.fillna(column_means)
    train = train[train['Gender'] == -1]
    train.to_pickle('Local_F_mean_imputation_multi_PUMA_cat.pkl')
    ########## mean imputation ###########

    ########## RF imputation ###########
    # train_df = train_df.drop(['Gang_Affiliated'], axis=1)
    # train = train_df[train_df['Gender'] == -1]
    # train.to_pickle('Local_F_missForest.pkl')
    ########## RF imputation ###########

    print('#'*50 + ' end ' + '#'*50)


if __name__ == '__main__':
    main()
