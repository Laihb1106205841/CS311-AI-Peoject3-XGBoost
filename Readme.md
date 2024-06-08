# Predicting Adult Census Income Using XGBoost Gradient Boosted Trees System
### Haibin Lai 赖海斌
### 12211612

<i class="fab fa-python"></i>

![img.png](Img/XGBoost1.png)

In this project, We try to predict Adult Census Income Dataset by using XGBoost Gradient Boosted Trees System.


This directory contains 3 main programs:

* `AutogluonProcess.ipynb` & `AutogluonProcess.py`: These files are used to run Autogluon and help us decide which model to use.

* `XGBoost.ipynb`: This file is used to run XGBoost.

* `Visualization.ipynb`: This file is for data visualization.


### Running Preparation
**Needed Python Version:**
 <=3.11 (package autogluon can't support python 3.12!) 
 My computer runs well in 3.11

**Needed Package:**

| Library    | Version   | Notes                   |
|------------|-----------|-------------------------|
| autogluon  | 1.1.0     | 自动化机器学习包                |
| pandas     | 2.2.2     | 处理数据                    |
| warnings   | in Python | -                       |
| matplotlib | 3.9.0     | 画图                      |
| seaborn    | 0.13.2    | 数据可视化                   |
| sklearn    | 1.4.0     | 决策树框架 (1.5.0 tested OK) |
| xgboost    | 2.0.3     | 提供 XGBoost 分类器          |



### Using xgboost:
XGBoost: eXtreme Gradient Boosting library.
Contributors: https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md 

**Installing**

Pip 21.3+ is required

```commandline
pip install xgboost
```

Installing autogluon (tips: install time is a bit long! And we don't fully need it!)
```commandline
pip install autogluon
```

This project's Github website:
https://github.com/Laihb1106205841/CS311-AI-Peoject3-XGBoost.git

XGBoost Introduction:
https://xgboost.readthedocs.io/en/stable/tutorials/model.html

XGBoost Github website:
https://github.com/dmlc/xgboost?tab=security-ov-file
