#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: 狂小腾
# @Date: 2022/5/14 13:51

"""
kaggle实战---泰坦尼克号
根据乘客信息，预测该名乘客是否被获救，0：表示未获救，1：表示获救
"""

import pandas as pd
import numpy as np
import seaborn as sns

# 1. 读取训练集和测试集数据
train = pd.read_csv('./datasets/train.csv')
test = pd.read_csv('./datasets/test.csv')

# 2. 数据探索分析
print(train.shape)  # (891, 12)
print(train.head())
print(train.info())
print(test.shape)  # (418, 11)
print(test.head())
print(test.info())

# 合并train和test 形成一个数据集 进行统一数据清理、特征构建
dataset = pd.concat([train, test], axis=0).reset_index(drop=True)  # drop=True 把原来的索引index列去掉
print(dataset.shape)  # (1309, 12)
print(dataset.head())

# 用NaN填充空缺的地方
dataset = dataset.fillna(np.nan)
# 检查Null值，统计一些数据集中有哪些字段存在NaN值
print(dataset.isnull().sum())
print(dataset.dtypes)  # 查看数据集类型
# 查看train训练集的描述
print(train.describe())
# 查看test的描述
print(test.describe())

# 3. 数据可视化展示
# 3.1 相关性矩阵 查看这些列之间是否存在相关性
# 列：SibSp, Parch, Age, Fare, Survived
g = sns.heatmap(train[['Survived', 'SibSp', 'Parch', 'Age', 'Fare']].corr(), annot=True, cmap='coolwarm', fmt='.2f')

