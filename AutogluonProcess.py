###########################################################################
# Try to implement different Model on the dataset by using autogluon

# 如果你没有看AutogluonProcess.ipynb，我推荐你先去看那个文件里的文档
# please look at file AutogluonProcess.ipynb first

# Input:    data/traindata.csv, data/trainlabel.txt
# Output:   data/traindata_preprocessed.csv predict/Autogluon_predictions.csv
# Running logs:
# /logs/AutoluonLog.txt
###########################################################################

import autogluon as ag
import pandas as pd
from autogluon.tabular import TabularDataset

import warnings
warnings.filterwarnings('ignore')

# -------------------------- Preprocessing ----------------------------- #
# 读取 train.csv 文件
train_df = pd.read_csv("data/traindata.csv")

# 读取trainlabel.txt文件
with open('data/trainlabel.txt', 'r') as file:
    lines = file.readlines()

# 解析文本数据并创建DataFrame
data = {'Label': [line.strip() for line in lines]}
df = pd.DataFrame(data)

# 将DataFrame保存为CSV文件
df.to_csv('data/trainlabel.csv', index=False)

print("trainlabel.txt 已成功转换为 trainlabel.csv")
# 读取 data.csv 文件，假设数据列名为 'income'
data_df = pd.read_csv("data/trainlabel.csv")

# 将 'income' 数据列添加到 train_df 中
train_df['Income'] = data_df

# 将修改后的 train_df 保存为新的 CSV 文件
train_df.to_csv("data/traindata_preprocessed.csv", index=False)

# loading
train_data = pd.read_csv('data/traindata_preprocessed.csv')
test_data = pd.read_csv('data/testdata.csv')


# ---------------------------- Training ------------------------------ #
label_column = 'Income'

# training
predictor = ag.tabular.TabularPredictor(label=label_column, problem_type='binary',
                                        verbosity=4, path='AutogluonModels/Model')
# , visualizer='tensorboard'

predictor.fit(train_data=train_data)


# ---------------------------- Training Result ----------------------- #
results = predictor.fit_summary(show_plot=True)
# leaderboard = predictor.leaderboard(test_data)   #预测结果看板
lboard = predictor.leaderboard()
print(lboard)


# ---------------------------- Predicting ---------------------------- #
# predicting 使用AutoGluon预测测试集数据
test_predictions = predictor.predict(test_data)
print(test_predictions)

# 将测试集原本的标签列和预测之后的标签列合并为一个DataFrame
test_result = pd.concat([test_data, pd.Series(test_predictions, name='Predictions')], axis=1)

# 输出测试集原本的标签列和预测之后的标签列
print("测试集原本的标签列和预测之后的标签列：")
print(test_result)

# 将预测结果保存到 CSV 文件中
test_result.to_csv('predict/Autogluon_predictions.csv', index=False)

print("Predictions saved successfully to 'predict/Autogluon_predictions.csv'")

# 打开一个文本文件，并将 DataFrame 中的 'predictions' 列数据写入文件
with open('predict/Autogluon_predictions.txt', 'w') as f:
    # 遍历 'predictions' 列中的每个值，并将其写入文件
    for value in test_result['Predictions']:
        f.write(str(value) + '\n')
