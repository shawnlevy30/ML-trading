import os
import json
import pandas as pd
import numpy as np
import pandas_ta as ta
import talib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

pandas_ta_indicators = []
talib_indicators = []

def main():
    print("=== ML model evaluation script started===\n")

    data_path = #put data file here
    results_path = '/Users/shawnlevy/Desktop/BootCamp/ML/results'
    os.makedirs(results_path, exist_ok = True)
    print(f"data path: {data_path}")
    print(f"results will be saved to {results_path}\n")

    print ("loading data...")
    df = pd.read_csv(data_path, parse_dates = ['datetime'])
    print(f"Data loaded successfully. Shape {df.shape}\n")

    print("calculating technical indicators...")

#compute pandas_ta indicators
    df['rsi'] = ta.rsi(df['close'])
    macd = ta.macd(df['close'])
    df['macd'] = macd['MACD_12_26_9']
    df['ema'] = ta.ema(df['close'])

#compute talib indicators
    df['adx'] = talib.ADX(df['high'],df['low'], df['close'])
    df['cci'] = talib.CCI(df['high'],df['low'], df['close'])
    df['roc'] = talib.ROC(df['close'])

    print("technical indicators calculated. current columns:")
    print(df.columns.tolist(),"\n")

## Prepare the Dataset with Lagged Features**
    print ("preparing the dataset with lagged features to prevent data leakage")

# Shift the target variable to prevent data leakage
    df['close_future'] = df['close'].shift(-1)

# Drop the last row as it will have NaN in 'close_future'    
    df = df[:-1]

# Drop rows with NaN values resulting from indicator calculations and shifting
    df.dropna(inplace=True)
    print(f"dropped rows with NaN values. New shape: {df.shape}")

#define features and target 
    features = ['rsi', 'macd', 'ema', 'adx', 'cci', 'roc']
    X = df[features]
    y = df['close_future']

    #split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle = False
    )

    print('data split in to training and testing sets.')
    print(f"training set length: {len(X_train)}, Testing set length: {len(X_test)}\n")

    print("defining machine learning models...")
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(n_estimators = 100, random_state = 42),
        'XGBRegressor': XGBRegressor(n_estimators =100,  random_state = 42, verbosity = 0),
        'SVR': SVR(),
        'MLPRegressor': MLPRegressor(hidden_layer_sizes=(50,50), max_iter = 500, random_state=42)    
    }
    print(f"models defind: {list(models.keys())}\n")

    print("training and evaluating models...\n")
    results= {}

    all_feature_importances = {}

    for name, model in tqdm(models.items(), desc = "training models"):
        tqdm.write(f"---\nTraining {name}...")

        #set verbosity for applicable models
        if name == 'RandomForestRegressor':
            model.set_params(verbose=0)
        elif name == 'XGBRegressor':
            model.set_params(verbosity=0)
        
        model.fit(X_train, y_train)
        tqdm.write(f"{name} finished training")

        #make predictions
        y_pred = model.predict(X_test)

        #calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[name] = {'MSE':mse, 'R2':r2}
        tqdm.write(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}\n")

        #feature importance extraction

        if name in ['RandomForestRegressor', 'XGBRegressor']:
            #built in feature importance
            importances = model.feature_importances_
            all_feature_importances[name] = importances
        elif name == 'LinearRegression':
            importances = np.abs(model.coef_)
            all_feature_importances[name] = importances
        else:
            print(f"calculating permutation importance for {name}...")
            perm_importance = permutation_importance(
                model, X_test, y_test, n_repeats = 10, random_state = 42, scoring = 'r2'
            )
            all_feature_importances[name] = perm_importance.importances_mean

    print("saving model performance results...")
    results_df = pd.DataFrame.from_dict(results, orient = 'index').reset_index()
    results_df = results_df.rename(columns= {'index': 'Model'})

    results_file = os.path.join(results_path, 'model_performance.csv')
    results_df.to_csv(results_file, index=False)
    print(f"model performance saved to : results_file\n")

    print("analyzing feature importance...\n")

    feature_importance_df= pd.DataFrame({'Feature': features})

    for model_name, importances in all_feature_importances.items():
        feature_importance_df[model_name] = importances

    for model_name in all_feature_importances.keys():
        feature_importance_df = feature_importance_df.sort_values(by=model_name, ascending= False)

    fi_file= os.path.join(results_path, 'feature_importance.csv')
    feature_importance_df.to_csv(fi_file, index=False)
    print(f"feature importance saved to: {fi_file}\n")

    print("=== Summary of model performance===")
    print(results_df.to_string(index=False))
    print("\n ===feature importance ===")
    print(feature_importance_df)

    best_model_name = results_df.sort_values(by= 'R2', ascending = False).iloc[0]['Model']
    best_model = models[best_model_name]  
    best_y_pred = best_model.predict(X_test)

    plt.figure(figsize = (12,6)) 
    plt.plot(y_test.values[:1000], label = 'Actual', color = 'blue')
    plt.plot(best_y_pred[:1000], label=f'Predicted({best_model_name})', color = 'red') 
    plt.legend()
    plt.title(f'Actual vs Predicted Close Prices - {best_model_name}, (first 1000 samples)')
    plt.xlabel('Sample')
    plt.ylabel('Close Price')
    plt.tight_layout()
    plot_file = os.path.join(results_path, f'actual_vs_predicted_{best_model_name}.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"\nActual vs Predicted plot saved to: {plot_file}")
    

    #optional - plotting feature importance for each model

    print("\nPLotting feature importances for each model...")
    for model_name in all_feature_importances.keys():
        plt.figure(figsize=(8,6))
        if model_name in ['RandomForestRegressor', 'XGBRegressor', 'LinearRegression']:
            importances = all_feature_importances[model_name]
            plt.barh(features, importances)
            plt.xlabel('Feature Importance')
            plt.title (f' Feature importance - {model_name}')
        else:
            importances = all_feature_importances[model_name]
            plt.barh(features, importances)
            plt.xlabel('Permutation importance')
            plt.title(f'permutation feature importance - {model_name}')
        plt.tight_layout()
        fi_plot_file=os.path.join(results_path, f'feature_importance_{model_name}.png')
        plt.savefig(fi_plot_file)
        plt.close()
        print(f"Feature importance plot saved to: {fi_plot_file}")
    
    print("\n=== ML model Evaluation complete===")

if __name__ == "__main__":
    main()