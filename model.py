import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from utils import eval, visual, save
import warnings
warnings.filterwarnings('ignore')

def Model(model, X_train, X_test, y_train, y_test, model_name, save_dir):
    model.fit(X_train, y_train)
#     print(model.best_params_)
    train_y_pred = model.predict(X_train)
    test_y_pred = model.predict(X_test)
    
    # train
    r2score, mse, rmse, error_ratio, mae = eval(y_train, train_y_pred)
    print(f'{model_name} train: \n r2: {r2score}  rmse: {rmse}  mse:{mse} error_ratio:{error_ratio} mae:{mae}')
    #test
    r2score, mse, rmse, error_ratio, mae = eval(y_test, test_y_pred)
    print(f'{model_name} test: \n r2: {r2score}  rmse: {rmse} mse:{mse} error_ratio:{error_ratio} mae:{mae}')

    save(model_name, save_dir, train_y_pred.flatten(), test_y_pred.flatten(), y_train.flatten(), y_test.flatten())
    
    return model

