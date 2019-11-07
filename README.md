# 语音性别识别技术报告

> 姓名： 梁新年
>
> 学号：BY1906024



## 代码运行

### 环境 requirements

- pandas
- Scikit-learn
- Xlrd (读取xlsx文件需要)
- xgboost
- numpy

### 代码运行



## 任务简介



## 模型原理

### 决策树

### 集成学习 AdaBoost

### GDBoost

### XGBoost



## 调参方法与数据选择

参数调整采用格点搜索的方法对如下几个参数进行搜索：

- 'booster': ['gbtree', 'dart'],
- 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]
- 'max_depth': [3, 4, 5, 6, 7]
- 'n_estimators': [100, 200, 500, 1000, 1500]

搜索给出的最优参数为：

- booster = gbtree
- Gamma = 0.001
- max_depth = 6
- n_estimators = 1500



## 结果与排名 20191107

![image-20191107191611966](/Users/xinnianliang/Library/Application Support/typora-user-images/image-20191107191611966.png)

最终效果为0.993838.