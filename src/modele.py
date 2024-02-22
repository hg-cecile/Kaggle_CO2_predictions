import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import shapiro

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from category_encoders import CatBoostEncoder, LeaveOneOutEncoder, TargetEncoder

import lightgbm as lgb

import preprocessing as prepro
import modele

from typing import List, Set, Tuple


def xgboost_prediction(X_train, X_validation, y_train, y_validation) :

    dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
    dval_reg = xgb.DMatrix(X_validation, y_validation, enable_categorical=True)

    # hyperparamètres
    params = {"objective": "reg:squarederror", 
          "max_depth":30,    
          "learning_rate":0.005,   
          "min_split_loss" : 10,   
          "random_state" : 42,   
          "min_child_weight"  : 1,  
          "max_delta_step" : 0,  
          "subsample" : 1,  
          "max_leaves" : 0,   
          "max_bin" : 800,  
          "num_parallel_tree" : 1,   
          "reg_alpha" : 0.8,  
          "reg_lambda" : 0.2,  
          "colsample_bytree":0.85,  
          "colsample_bynode" : 0.85,  
          "colsample_bylevel" : 0.85,
          'eval_metric': "mae"}

    evals = [(dtrain_reg, "train"), (dval_reg, "validation")]

    n = 3000
    model = xgb.train(
        params=params,
    dtrain=dtrain_reg,
    num_boost_round=n,
    evals=evals,
    verbose_eval=200,
    early_stopping_rounds = 20 # pour limiter overfitting
    )

    y_pred = model.predict(dval_reg)

    mae = mean_absolute_error(y_validation, y_pred)

    print(f"MAE : {mae}")

    return model

