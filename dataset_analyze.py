import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib as jb
from pandas._libs.missing import NA
from sklearn.linear_model import LogisticRegression
import sys

cardioVascularDF = pd.read_csv('Dataset1 - cardio-vascular dataset.csv')
# print(cardioVascularDF.head(20))
cardioVascularDF.info()
cardioVascularDF.describe().transpose()
corr = cardioVascularDF.corr()
print(corr)
sns.heatmap(
    corr,
    xticklabels=corr.columns,
    yticklabels=corr.columns
)
# check for null values
print(sum(cardioVascularDF.isnull().sum()))
# replace Nan values with the mean column value
cardioVascularDF.fillna(cardioVascularDF.mean, inplace=True)
print(sum(cardioVascularDF.isnull().sum()))
print(sys.getrecursionlimit())
sys.setrecursionlimit(1500)
# check if Nan values are replaced in a random column
print((cardioVascularDF['education'].isnull().sum()))
