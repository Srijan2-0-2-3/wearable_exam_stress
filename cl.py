import time
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from avalanche.benchmarks.generators import dataset_benchmark, nc_benchmark
from avalanche.training.supervised import Naive, Cumulative, LwF, EWC, JointTraining, GEM, Replay
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, timing_metrics, \
    cpu_usage_metrics, disk_usage_metrics, gpu_usage_metrics
from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
import pickle
import torch.nn as nn
import torch

import warnings

warnings.filterwarnings("ignore")


class FeatureDataset(Dataset):
    def __init__(self, student, device='cpu'):
        super(FeatureDataset, self).__init__()
        df = pd.read_csv(f'{student}_midsem_dataset.csv', index_col=0)
        score = df['score']
        df = df.drop('score', axis=1)
        self.data = torch.tensor((np.array(df)), dtype=torch.float32).to(device)
        print(self.data.shape)
        # print(type(self.data))
        # score = pd.read_csv(f'{student}_Final.csv', header=None).iloc[0, 0]
        # target_data = np.full((len(df), 1), score)
        self.targets = torch.tensor(np.array(score), dtype=torch.long).squeeze().to(device)
        print(self.targets.shape)

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
        self.output_1 = nn.Linear(hidden_dim, out_dim)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)

        x = self.sigmoid(x)
        x = self.output_1(x)
        x = self.softmax(x)

        return x


students = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']
train_students = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
test_student = ['S10']
# train_dataset_final = []
# test_dataset_final = []
# train_dataset_midterm1 = []
# test_dataset_midterm1 = []
# train_dataset_midterm2 = []
# test_dataset_midterm2 = []
# exams = ['Final', 'Midterm 1', 'Midterm 2']
# print(len(df))
# df.to_csv(f'{student}_{exam}_dataset.csv')
# df_tf = tf.convert_to_tensor(df)
# print(df_tf.shape)
# score = pd.read_csv(f'{student}_{exam}.csv',header=None).iloc[0,0]
# print(score)
# fd = FeatureDataset(df_tf, score)
# print(fd)


input_dim = 7
hidden_dim = 14
output_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Classifier(input_dim, hidden_dim, output_dim)
scenario = dataset_benchmark([FeatureDataset(subject,device) for subject in train_students],
                             [FeatureDataset(subject,device) for subject in test_student])
# scenario = dataset_benchmark([FeatureDataset('S1', device)], [FeatureDataset('S1', device)])
tb_logger = TensorboardLogger()
text_logger = TextLogger(open('wearable_exam_stress_log.txt', 'a'))
int_logger = InteractiveLogger()

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(experience=True),
    gpu_usage_metrics(0, experience=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=[text_logger]
)

es = EarlyStoppingPlugin(patience=25, val_stream_name="train_stream")

results = []

strats = ['naive', 'offline', 'replay', 'cumulative', 'lwf', 'ewc', 'episodic']
strat = 'offline'
# for strat in strats:
if (strat == "naive"):
    print("Naive continual learning")
    strategy = Naive(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                     train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
elif (strat == "offline"):
    print("Offline learning")
    strategy = JointTraining(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                             train_epochs=10, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
elif (strat == "replay"):
    print("Replay training")
    strategy = Replay(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                      train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, mem_size=70,
                      train_mb_size=70)  # 25% of WESAD
elif (strat == "cumulative"):
    print("Cumulative continual learning")
    strategy = Cumulative(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(),
                          train_epochs=100, eval_every=1, plugins=[es], evaluator=eval_plugin, device=device)
elif (strat == "lwf"):
    print("LwF continual learning")
    strategy = LwF(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=100,
                   eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, alpha=0.5, temperature=1)
elif (strat == "ewc"):
    print("EWC continual learning")
    torch.backends.cudnn.enabled = False
    strategy = EWC(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=100,
                   eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, ewc_lambda=0.99)
elif (strat == "episodic"):
    print("Episodic continual learning")
    strategy = GEM(model, Adam(model.parameters(), lr=0.005, betas=(0.99, 0.99)), CrossEntropyLoss(), train_epochs=100,
                   eval_every=1, plugins=[es], evaluator=eval_plugin, device=device, patterns_per_exp=70)

i = 0

thisresults = []
for experience in scenario.train_stream:
    start = time.time()
    print(experience)
    print(start)
    res = strategy.train(experience)
    r = strategy.eval(scenario.test_stream)
    print(f"loss:{r['Loss_Exp/eval_phase/test_stream/Task000/Exp000']}")
    print(f"acc: {r['Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000'] * 100}",
          f"forg: {r['StreamForgetting/eval_phase/test_stream']}")
    thisresults.append({"loss": r["Loss_Exp/eval_phase/test_stream/Task000/Exp000"],
                        "acc": (float(r["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"]) * 100),
                        "forg": r["StreamForgetting/eval_phase/test_stream"],
                        "all": r})
    with open(f'{strat}_{i}_wearable_exam_stress.pkl', 'ab') as f:
        pickle.dump(thisresults, f)
results.append({"strategy": 'replay',
                "finalloss": r["Loss_Exp/eval_phase/test_stream/Task000/Exp000"],
                "finalacc": r["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp000"],
                "results": thisresults})
elapsed = time.time() - start
results.append({"time": elapsed})
with open("wearable_classifier_" + strat + "_results" + ".pkl", "wb") as outfile:
    pickle.dump(results, outfile)
