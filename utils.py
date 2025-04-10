import os
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
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
warnings.filterwarnings('ignore')

def error_ratio(y_test, y_pred):
    y_test = np.array(y_test).flatten()
    y_pred = np.array(y_pred)
    ratio = []
    for t, p in zip(y_test, y_pred):
        r = abs(p - t)/t
        ratio.append(r)
    return str(round(np.mean(ratio)*100, 2))+'%'

def eval(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    er = error_ratio(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return r2, mse, rmse, er, mae

def visual(model, model_name, train, test, train_label, test_label):
    # train
    pred_train = model.predict(train).flatten()
    pred_test = model.predict(test).flatten()

    # 决策树模型可视化
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False  

    plt.figure(figsize=(10, 6))
    y_train = train_label.flatten()
    y_test = test_label.flatten()
    
    sns.scatterplot(x=y_train, y = pred_train)
    sns.scatterplot(x=y_test, y=pred_test)
    
    plt.title(f'{model_name}')
    plt.xlabel('GroundTruth')
    plt.ylabel('Predict')

    plt.plot([0, 3], [0, 3], linestyle='--', color='red', linewidth=2, label='Perfect Curves')
    plt.legend()
    plt.show()
    
def save(file_name, dir_name, pred_train, pred_test, label_train, label_test):
    if not os.path.exists(f'result/{dir_name}'):
        os.makedirs(f'result/{dir_name}')
    result_df = pd.DataFrame(columns=['train_pred','train_truth'])
    result_df['train_pred'] = pred_train
    result_df['train_truth'] = label_train
    result_df.to_csv(f'result/{dir_name}/{file_name}_train.csv', index=False)
    result_df1 = pd.DataFrame(columns=['test_pred','test_truth'])
    result_df1['test_pred'] = pred_test
    result_df1['test_truth'] = label_test
    result_df1.to_csv(f'result/{dir_name}/{file_name}_test.csv', index=False)
    print('save result successful！')
    
def save_model(model, model_name):
    with open(f'model/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f'save {model_name} model successful')