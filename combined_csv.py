import pandas as pd
import tensorflow as tf
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
path = "a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0\\Data\\Data\\"

students = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']

exams = ['Final', 'Midterm 1', 'Midterm 2']

features = ['BVP', 'EDA', 'HR', 'TEMP']


# import os
#
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print(files)
# print(os.listdir(path))
class FeatureDataset(Dataset):
    def __init__(self, student, df, score):
        self.student = student
        self.data = torch.tensor((np.array(df)), dtype=torch.float32)
        self.targets = torch.tensor(score, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # print(self.targets[idx])
        return self.data[idx], self.targets[idx]


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Classifier, self).__init__()
        # print('init')
        self.hidden = nn.Linear(in_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)

        x = self.sigmoid(x)

        x = self.output(x)

        x = self.softmax(x)

        return x



# for student in students:
#     for exam in exams:
#         df = pd.read_csv(f'{path}{student}\\{exam}\\ACC.csv', header=None)
#         initial_timestamp = df.iloc[0, 0]
#         # print(initial_timestamp)
#         sample_rate = df.iloc[1, 0]
#         # print(sample_rate)
#         timestamps = [initial_timestamp + i / sample_rate for i in range(len(df) - 3)]
#         df = df.loc[3:, ]
#
#         df['Timestamp'] = timestamps
#         df.columns = ['x_acc', 'y_acc', 'z_acc', 'Timestamp']
#         # print(df.head())
#         for feature in features:
#             df1 = pd.read_csv(f'{path}{student}\\{exam}\\{feature}.csv', header=None)
#             # print(df1.head())
#             # print(df1.shape)
#             initial_timestamp = df1.iloc[0, 0]
#             # print(initial_timestamp)
#             sample_rate = df1.iloc[1, 0]
#             # print(sample_rate)
#             timestamps = [initial_timestamp + i / sample_rate for i in range(len(df1) - 3)]
#             # print(len(timestamps))
#
#             df1 = df1.loc[3:, ]
#
#             df1['Timestamp'] = timestamps
#             # print(df1.head())
#             if feature == 'BVP':
#                 df1.columns = ['bvp', 'Timestamp']
#             elif feature == 'EDA':
#                 df1.columns = ['eda', 'Timestamp']
#             elif feature == 'HR':
#                 df1.columns = ['hr', 'Timestamp']
#             elif feature == 'TEMP':
#                 df1.columns = ['temp_c', 'Timestamp']
#             df = df.merge(df1, on='Timestamp', how='inner')
#         # df.to_csv(f'{student}_{exam}.csv')
#         # print(df.shape)
#         # apply normalization
#         df = df.drop(['Timestamp'], axis=1)
#         for column in df.columns:
#             df[column] = (df[column] - df[column].mean()) / df[column].std()
#         # print(len(df))
#         # arr = df.to_numpy()
#         # print(arr.shape)
#         # df_tf = tf.convert_to_tensor(df)
#         # print(type(df_tf))
#         # print(df_tf.shape)
#         # print(df.columns)
#         # print(df.head())
#         df_score = pd.read_csv(f'{student}_{exam}.csv', header=None)
#         score = df_score.iloc[0, 0]
#         # print(score)
#
#         target_data = np.full((len(df), 1), score)
#         df['score'] = target_data
#         df.to_csv(f'{student}_{exam}_ds.csv')


