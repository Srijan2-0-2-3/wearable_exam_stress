import pandas as pd
import tensorflow as tf
from torch.utils.data import Dataset
import torch.nn as nn
import torch

path = "a-wearable-exam-stress-dataset-for-predicting-cognitive-performance-in-real-world-settings-1.0.0\\Data\\Data\\"
alternate_path = 'Data/'
students = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']

exams = ['Final', 'Midterm 1', 'Midterm 2']

features = ['BVP', 'EDA', 'HR', 'TEMP']


class DataSplit(object):
    def __init__(self, training_set, testing_set):
        self.training_set = training_set
        self.testing_set = testing_set


class FeatureDataset(Dataset):

    def __init__(self, student_id, df, score):
        self.student_id = student_id
        self.data = torch.tensor(df,dtype=torch.float32)
        self.target = torch.tensor(score, dtype=torch.long)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        return self.data[item], self.target[item]


df = pd.read_csv(f'S1_Final.csv', header=None)
score = df.iloc[0,0]
df_dataset = pd.read_csv(f'S1_Final_dataset.csv')

fd = FeatureDataset('S1',df_dataset,score)
print(fd.__len__())

# class Classifier(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super(Classifier, self).__init__()
#         # print('init')
#         self.hidden = nn.Linear(in_dim, hidden_dim)
#         self.output = nn.Linear(hidden_dim, out_dim)
#         self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.hidden(x)
#
#         x = self.sigmoid(x)
#
#         x = self.output(x)
#
#         x = self.softmax(x)
#
#         return x
#
#
# def create_leave_one_out():
#     splits = []
#
#     for index in range(len(students)):
#         training_set = students.copy()
#         testing_set = [training_set.pop(index)]
#
#         splits.append(DataSplit(training_set=training_set, testing_set=testing_set))
#
#     return splits
#
#
# # for student in students:
# #     for exam in exams:
# #         df = pd.read_csv(f'{alternate_path}{student}/{exam}/ACC.csv', header=None)
# #         initial_timestamp = df.iloc[0, 0]
# #         # print(initial_timestamp)
# #         sample_rate = df.iloc[1, 0]
# #         # print(sample_rate)
# #         timestamps = [initial_timestamp + i / sample_rate for i in range(len(df) - 3)]
# #         df = df.loc[3:, ]
# #
# #         df['Timestamp'] = timestamps
# #         df.columns = ['x_acc', 'y_acc', 'z_acc', 'Timestamp']
# #         for feature in features:
# #             df1 = pd.read_csv(f'{alternate_path}{student}/{exam}/{feature}.csv', header=None)
# #             initial_timestamp = df1.iloc[0, 0]
# #             sample_rate = df1.iloc[1, 0]
# #             timestamps = [initial_timestamp + i / sample_rate for i in range(len(df1) - 3)]
# #
# #             df1 = df1.loc[3:, ]
# #
# #             df1['Timestamp'] = timestamps
# #             if feature == 'BVP':
# #                 df1.columns = ['bvp', 'Timestamp']
# #             elif feature == 'EDA':
# #                 df1.columns = ['eda', 'Timestamp']
# #             elif feature == 'HR':
# #                 df1.columns = ['hr', 'Timestamp']
# #             elif feature == 'TEMP':
# #                 df1.columns = ['temp_c', 'Timestamp']
# #             df = df.merge(df1, on='Timestamp', how='inner')
# #         # applying normalization
# #         df = df.drop('Timestamp', axis=1)
# #         print(df.columns)
# #         # columns = ['x_acc', 'y_acc', 'z_acc', 'bvp', 'eda', 'hr', 'temp_c']
# #         for column in df.columns:
# #             df[column] = (df[column] - df[column].mean()) / df[column].std()
# #         print(len(df))
# #         df.to_csv(f'{student}_{exam}_dataset.csv')
# #         # df_tf = tf.convert_to_tensor(df)
# #         # print(df_tf.shape)
# #         # score = pd.read_csv(f'{student}_{exam}.csv',header=None).iloc[0,0]
# #         # print(score)
# #         # fd = FeatureDataset(df_tf, score)
# #         # print(fd)
# split = create_leave_one_out()[0]
#
# # for student in split.training_set:
# #     for exam in exams:
# #         if exam != 'Final':
#
#
# input_dim = 7
# hidden_dim = 14
# output_dim_midterm = 100
# output_dim_final = 200
