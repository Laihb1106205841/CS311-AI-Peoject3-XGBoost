# Predicting Adult Census Income Using XGBoost Gradient Boosted Trees System
### Haibin Lai 赖海斌 12211612

<i class="fab fa-python"></i>

![img.png](Img/XGBoost1.png)

In this project, We try to predict Adult Census Income Dataset by using XGBoost Gradient Boosted Trees System.


This directory contains 3 main programs:

* `AutogluonProcess.ipynb` & `AutogluonProcess.py`: These files are used to run Autogluon and help us decide which model to use.

* `XGBoost.ipynb`: This file is used to run XGBoost.

* `Visualization.ipynb`: This file is for data visualization.


### Running Preparation
**Needed Python Version:**

 ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !

**python <=3.11**
(package autogluon can't support python 3.12!) 

 ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !

My computer runs well in 3.11


**Needed Package:**

The following version work well in Anaconda:

| Library    | Version   | Notes                   |
|------------|-----------|-------------------------|
| autogluon  | 1.1.0     | 自动化机器学习包                |
| pandas     | 2.2.1     | 处理数据                    |
| warnings   | in Python | -                       |
| matplotlib | 3.8.4     | 画图                      |
| seaborn    | 0.12.0    | 数据可视化                   |
| sklearn    | 1.4.0     | 决策树框架 (1.5.0 tested OK) |
| xgboost    | 2.0.3     | 提供 XGBoost 分类器          |
| jupyter    | 1.0.0     |                         |



### Using xgboost:
XGBoost: eXtreme Gradient Boosting library.
Contributors: https://github.com/dmlc/xgboost/blob/master/CONTRIBUTORS.md 

**Installing**

Pip 21.3+ is required

```commandline
pip install xgboost
```

**Installing autogluon**

(tips: install time is a bit long! And we don't fully need it!)

这个包有很多版本限制，它可能会改掉一些下载的东西（比如它会卸载scikit-learn 1.4.2，然后重新安装个版本）要依赖很多包（比如torch），文件非常大，并且需要管理员权限来安装，我们只在Autogluon文件中进行了跑，
并且最终提交没有依赖那两个文件，而是XGBoost文件，如果跑不起来也没问题。
另外很神奇的是，这个包的预测跟它的版本有关
```commandline
pip install autogluon
```

### Useful website

This project's Github website:
https://github.com/Laihb1106205841/CS311-AI-Peoject3-XGBoost.git

XGBoost Introduction:
https://xgboost.readthedocs.io/en/stable/tutorials/model.html

XGBoost Github website:
https://github.com/dmlc/xgboost?tab=security-ov-file
