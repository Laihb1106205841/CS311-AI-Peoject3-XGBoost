import autogluon as ag
import pandas as pd
from autogluon.tabular import TabularDataset
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np


# loading
train_data = pd.read_csv('data/traindata1.csv')
test_data = pd.read_csv('data/testdata1.csv')

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