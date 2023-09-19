import pandas as pd
import tensorflow as tf
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn

path = "a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0/Data/"

students = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']

exams = ['Midterm 1', 'Midterm 2']

features = ['BVP', 'EDA', 'HR', 'TEMP']
common_columns = ['Timestamp', 'x_acc', 'y_acc', 'z_acc', 'bvp', 'eda', 'hr', 'temp_c']

for student in students:
    print(student)
    df_main = pd.DataFrame(columns=common_columns + ['score'])
    for exam in exams:
        print(exam)
        # df = pd.DataFrame(columns=[])
        df = pd.read_csv(f'{path}{student}/{exam}/ACC.csv', header=None)
        initial_timestamp = df.iloc[0, 0]
        sample_rate = df.iloc[1, 0]
        timestamps = [initial_timestamp + i / sample_rate for i in range(len(df) - 3)]
        df = df.loc[3:, ]

        df['Timestamp'] = timestamps
        df.columns = ['x_acc', 'y_acc', 'z_acc', 'Timestamp']
        for feature in features:
            df1 = pd.read_csv(f'{path}{student}/{exam}/{feature}.csv', header=None)
            initial_timestamp = df1.iloc[0, 0]
            sample_rate = df1.iloc[1, 0]
            timestamps = [initial_timestamp + i / sample_rate for i in range(len(df1) - 3)]
            df1 = df1.loc[3:, ]

            df1['Timestamp'] = timestamps
            if feature == 'BVP':
                df1.columns = ['bvp', 'Timestamp']
            elif feature == 'EDA':
                df1.columns = ['eda', 'Timestamp']
            elif feature == 'HR':
                df1.columns = ['hr', 'Timestamp']
            elif feature == 'TEMP':
                df1.columns = ['temp_c', 'Timestamp']
            df = df.merge(df1, on='Timestamp', how='inner')
        df = df[common_columns]
        df['score'] = np.nan
        for column in df.columns:
            df[column] = (df[column] - df[column].mean()) / df[column].std()
        df_score = pd.read_csv(f'{student}_{exam}.csv', header=None)
        score = int(df_score.iloc[0, 0])

        df['score'] = np.full((len(df), 1), score)
        print(df.shape)
        df_main = pd.concat([df, df_main], ignore_index=True)

    df_main = df_main.drop(['Timestamp'], axis=1)
    print(df_main.shape)
    df_main.to_csv(f'{student}_midsem_dataset.csv')
