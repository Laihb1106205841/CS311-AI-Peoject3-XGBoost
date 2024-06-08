###########################################################################
# Try to implement different Model on the dataset by using autogluon

# 如果你没有看AutogluonProcess.ipynb，我推荐你先去看那个文件里的文档
# please look at file AutogluonProcess.ipynb first
###########################################################################

import autogluon as ag
import pandas as pd
from autogluon.tabular import TabularDataset

import warnings
warnings.filterwarnings('ignore')

# -------------------------- Preprocessing ----------------------------- #
# 读取 train.csv 文件
train_df = pd.read_csv("data/traindata.csv")

# 读取 data.csv 文件，假设数据列名为 'income'
data_df = pd.read_csv("data/trainlabel.csv")

# 将 'income' 数据列添加到 train_df 中
train_df['Income'] = data_df['Income']

# 将修改后的 train_df 保存为新的 CSV 文件
train_df.to_csv("data/traindata_preprocessed.csv", index=False)

# loading
train_data = pd.read_csv('data/traindata_preprocessed.csv')
test_data = pd.read_csv('data/testdata.csv')


# ---------------------------- Training ------------------------------ #
label_column = 'Income'

# training
predictor = ag.tabular.TabularPredictor(label=label_column, problem_type='binary', verbosity=4)
# , visualizer='tensorboard'

predictor.fit(train_data=train_data)


# ---------------------------- Training Result ----------------------- #
results = predictor.fit_summary(show_plot=True)
# leaderboard = predictor.leaderboard(test_data)   #预测结果看板
lboard = predictor.leaderboard()
print(lboard)


# ---------------------------- Predicting ---------------------------- #
# predicting
performance = predictor.predict(test_data)
print(performance)

# 将结果保存到 CSV 文件
performance.to_csv('predict/performance_predictions.csv', index=False)

print("Predictions saved successfully to 'predict/performance_predictions.csv'")
