import pandas as pd
import numpy as np
import seaborn as sns
from statistics import mode
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

cardioVascularDF = pd.read_csv('Dataset1 - cardio-vascular dataset.csv', index_col=False)
cardioVascularDF.info()
cardioVascularDF.describe().transpose()

corr = cardioVascularDF.corr()
print(corr)
sns.heatmap(
    corr,
    xticklabels=corr.columns,
    yticklabels=corr.columns
)

print(cardioVascularDF.describe(include="all"))
cardioVascularDF["education"] = cardioVascularDF["education"].fillna(mode(cardioVascularDF["education"]))
cardioVascularDF["cigsPerDay"] = cardioVascularDF["cigsPerDay"].fillna(cardioVascularDF["cigsPerDay"].mean())
cardioVascularDF["BPMeds"] = cardioVascularDF["BPMeds"].fillna(mode(cardioVascularDF["BPMeds"]))
cardioVascularDF["totChol"] = cardioVascularDF["totChol"].fillna(cardioVascularDF["totChol"].mean())
cardioVascularDF["glucose"] = cardioVascularDF["glucose"].fillna(cardioVascularDF["glucose"].mean())
cardioVascularDF["BMI"] = cardioVascularDF["BMI"].fillna(cardioVascularDF["BMI"].mean())
cardioVascularDF["heartRate"] = cardioVascularDF["heartRate"].fillna(cardioVascularDF["heartRate"].mean())

# check if there are still Nan values in the detected columns
arr = np.asarray(cardioVascularDF['glucose'])
print(arr[14])
print("sum of nan values " + str(cardioVascularDF.isna().sum()))
a = False
for col in cardioVascularDF.columns:
    if cardioVascularDF[col].isnull().any():
        a = True
if a:
    print('null values in the dataset')

print('nan values   ' + str(cardioVascularDF.isna().sum()))

data = pd.DataFrame(cardioVascularDF.values)
X_features = cardioVascularDF.drop(["TenYearCHD"], axis=1)
Y_features = cardioVascularDF["TenYearCHD"]

bestFeatures = SelectKBest(score_func=chi2, k=10)
fit = bestFeatures.fit(X_features, Y_features)
dataScore = pd.DataFrame(fit.scores_)
dataColumns = pd.DataFrame(X_features.columns)
featureScore = pd.concat([dataColumns, dataScore], axis=1)
featureScore.columns = ["feature", "score"]
print(featureScore)
