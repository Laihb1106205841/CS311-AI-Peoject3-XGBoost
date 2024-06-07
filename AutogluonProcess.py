import autogluon as ag
import pandas as pd
from autogluon.tabular import TabularDataset
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np


# 读取 train.csv 文件
train_df = pd.read_csv("data/traindata.csv")

# 读取 data.csv 文件，假设数据列名为 'income'
data_df = pd.read_csv("data/trainlabel.csv")

# 将 'income' 数据列添加到 train_df 中
train_df['income'] = data_df['Income']

# 将修改后的 train_df 保存为新的 CSV 文件
train_df.to_csv("updated_train.csv", index=False)

#
#
# loading
train_data = pd.read_csv('data/traindata.csv')
test_data = pd.read_csv('data/testdata.csv')

test_data.head()

label_column = 'Income'

# training
predictor = ag.tabular.TabularPredictor(label=label_column, problem_type='binary', verbosity=4)
# , visualizer='tensorboard'

predictor.fit(train_data=train_data)

# predicting
performance = predictor.predict(test_data)
print(performance)

results = predictor.fit_summary(show_plot=True)
# leaderboard = predictor.leaderboard(test_data)   #预测结果看板
lboard = predictor.leaderboard()