import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
# 读取训练和submit数据
train = pd.read_excel('dataset/raw_data/train.xlsx')
test = pd.read_excel('dataset/raw_data/test.xlsx')
print("训练集列名", train.columns)

def preprocess(df, is_train=True):
    # 预处理文件，并且创建模型的数据输入流
    # 删除不好的五个特征
    df = df.drop(['skew','kurt','mindom','maxdom', 'centroid'], axis=1)
    temp_df = df
    scaler=StandardScaler()
    Y = None
    if is_train:
        Y = temp_df['label'].as_matrix()
        temp_df = temp_df.drop('label',axis=1)
    scaled_df = scaler.fit_transform(temp_df)
    
    return scaled_df, Y

X, Y = preprocess(train)
# 划分训练和验证的数据集
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.15, random_state=42)

# 格点搜索
# params_dict={'max_depth':[3, 4, 5, 6, 7],'gamma':[0.001,0.01,0.1,1,10,100], 
#              'n_estimators': [100, 200, 500, 1000, 1500], 'learning_rate': [0.1],
#             'object':['binary:logistic'], 
#             'booster': ['gbtree', 'dart']}
# clf=GridSearchCV(estimator=XGBClassifier(), param_grid=params_dict, scoring='roc_auc', cv=5, n_jobs=-1)
# clf.fit(x_train, y_train)

# print(clf.best_params_)

# print(clf.best_score_)
# res = clf.predict(x_test)
# print(roc_auc_score(res, y_test))

# 模型，使用搜索出来的最优参数
clf = XGBClassifier(max_depth=6,
                    learning_rate=0.1,
                    n_estimators=1500,
                    verbosity=1,
                    objective='binary:logistic',
#                     booster='gbtree', # gbtree, gblinear or dart.
                    booster='gbtree',
                    n_jobs=-1,
                    gamma=0.001,
                   reg_alpha=0.5,
                   reg_lambda=0.5)
# 训练模型
clf.fit(x_train, y_train)

# 验证集评分
res = clf.predict(x_test)
print("验证集评分：", roc_auc_score(res, y_test))

# 生成提交文件
submit_x, _ = preprocess(test, False)
res = clf.predict_proba(submit_x)
print("生成submission.txt文件...")
with open('submission.txt', 'w') as f:
    for i in res:
        f.writelines("%.6f" % i[1])
        f.writelines('\n')

# 生成压缩文件
import zipfile
print("压缩submission.zip 文件")
with zipfile.ZipFile('submission.zip', 'w', zipfile.ZIP_DEFLATED) as f:
    f.write('submission.txt')
print("Ended!")