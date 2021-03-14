#Load packages
import pandas as pd
import os
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb


dirn = os.path.dirname(os.path.realpath(__file__))
train_csv = os.path.join(dirn, 'data', 'train.csv')

train_df = pd.read_csv(train_csv)

#Linear predictors
lin_preds = ['1stFlrSF', '2ndFlrSF']
for i in lin_preds:
    train_df.plot.scatter(i, 'SalePrice')
    
#Categorical predictors
cat_prefs = ['OverallQual', 'Neighborhood']
for i in cat_prefs:
    fig = train_df.boxplot('SalePrice', by = i, patch_artist = True)
    print(fig)
    
    
    
#Fit model
train_preds = train_df[['LotArea', 'YearBuilt', 'YearRemodAdd', 'OverallQual', 'OverallCond',
                        'Neighborhood', 'BldgType', 'HouseStyle', 'ExterQual',
                        'ExterCond', 'Street', 'LandSlope', 'RoofStyle', 'RoofMatl',
                        'Foundation', 'MSSubClass', 'MSZoning', 'LotArea', 
                        'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition',
                        'GarageCars', 'LotConfig', 'Condition1', 'Condition2',
                        'Functional', 'KitchenQual', 'TotRmsAbvGrd',
                        '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'Fence']]
cats = ['Neighborhood', 'BldgType', 'HouseStyle', 'ExterQual', 'ExterCond', 'Street',
        'LandSlope', 'RoofStyle', 'RoofMatl', 'Foundation', 'MSSubClass', 'MSZoning',
        'GarageType', 'PavedDrive', 'SaleType', 'SaleCondition', 'LotConfig',
        'Condition1', 'Condition2', 'Functional', 'KitchenQual', 'Fence']
for col in cats:
    dummies = pd.get_dummies(train_preds[col])
    train_preds[dummies.columns] = dummies
    
features = train_preds.drop(columns = cats)
feature_list = list(features.columns)
features = np.array(features)
labels = np.array(train_df[['SalePrice']])

lm = LinearRegression().fit(features, labels)
lm_preds = lm.predict(features)

rf = RandomForestRegressor(n_estimators = 500, random_state = 42)
rf.fit(features, labels)
xg_preds = rf.predict(features)

def mape(y_true, y_pred): 
     y_true, y_pred = np.array(y_true), np.array(y_pred)
     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(mape(labels, lm_preds))
print(mape(labels, xg_preds))