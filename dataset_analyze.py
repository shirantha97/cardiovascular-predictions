import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib as jb
# from pandas._libs.missing import NA
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
print('the sum null values ' + str(sum(cardioVascularDF.isnull().sum())))
print(sys.getrecursionlimit())
sys.setrecursionlimit(1500)
# check if Nan values are replaced in a random column
print((cardioVascularDF['education'].isnull().sum()))

dfTrain = cardioVascularDF[:3000]
dfTest = cardioVascularDF[3000:4000]
dfCheck = cardioVascularDF[4000:]

trainLbl = np.asarray(dfTrain['TenYearCHD'])
trainData = np.asarray(dfTrain.drop('TenYearCHD', 1))
# print(trainData)
testLbl = np.asarray(dfTest['TenYearCHD'])
testData = np.asarray(dfTest.drop('TenYearCHD', 1))
