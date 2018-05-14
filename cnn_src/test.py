import pickle

import numpy as np
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

feature, label, feature_test, label_test = pickle.load(open("temp.pkl", "rb"))
print(np.array(feature).shape, np.array(label).shape)

randomForestClassifier = RandomForestClassifier(n_estimators=100,
                                                max_depth=13,
                                                min_samples_split=110,
                                                min_samples_leaf=20,
                                                max_features='sqrt',
                                                oob_score=True,
                                                random_state=10).fit(feature, label)

xgboostClassifier = xgb.XGBClassifier(colsample_bytree=0.4,
                                      gamma=0.05,
                                      learning_rate=0.16,
                                      max_depth=6,
                                      n_estimators=85,
                                      reg_alpha=1.2,
                                      subsample=1,
                                      objective='binary:logistic',
                                      silent=False,
                                      nthread=8).fit(feature, label)

print("********************************** randomForestClassifier **********************************")
print(classification_report(label_test, randomForestClassifier.predict(feature_test)))
print("********************************** randomForestClassifier **********************************")

print("************************************* xgboostClassifier ************************************")
print(classification_report(label_test, xgboostClassifier.predict(feature_test)))
print("************************************* xgboostClassifier ************************************")

print("************************************* LogisticRegression ************************************")
print(classification_report(label_test, LogisticRegression().fit(feature, label).predict(feature_test)))
print("************************************* LogisticRegression ************************************")

