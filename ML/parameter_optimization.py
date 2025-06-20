import os
import pandas as pd
import numpy as np
import pandas_ta as ta
import talib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import warnings
from tqdm import tqdm
import itertools

warnings.filterwarnings('ignore')

def create_event_target(df, profit_pct=1.30, loss_pct=0.50, look_forward_bars=30):
    """
    Looks forward in time from each bar to see if a profit target is hit before a stop loss.
    
    Args:
        df (pd.DataFrame): The dataframe with 'open', 'high', 'low', 'close' prices.
        profit_pct (float): The target multiplier for profit (e.g., 1.30 for +30%).
        loss_pct (float): The target multiplier for loss (e.g., 0.50 for -50%).
        look_forward_bars (int): The maximum number of bars to look into the future.

    Returns:
        pd.Series: A series of 1s (profit target hit first) and 0s (loss hit or nothing happened).
    """
    n = len(df)
    target = np.zeros(n, dtype=int)
    
    for i in tqdm(range(n), desc="Creating Event Targets"):
        current_price = df['close'].iloc[i]
        profit_target_price = current_price * profit_pct
        loss_target_price = current_price * loss_pct

        # Define the window to look forward in
        forward_window = df.iloc[i+1 : i+1+look_forward_bars]
        if forward_window.empty:
            continue

        # Find the first index where profit or loss is hit
        profit_hit_index = (forward_window['high'] >= profit_target_price).idxmax()
        loss_hit_index = (forward_window['low'] <= loss_target_price).idxmax()

        # Determine if they were actually hit (idxmax returns first index if no True found)
        profit_was_hit = forward_window.loc[profit_hit_index, 'high'] >= profit_target_price
        loss_was_hit = forward_window.loc[loss_hit_index, 'low'] <= loss_target_price

        if profit_was_hit and loss_was_hit:
            # If both happen, see which came first
            if profit_hit_index <= loss_hit_index:
                target[i] = 1
        elif profit_was_hit:
            target[i] = 1
            
    return pd.Series(target, index=df.index)

def calculate_features(df):
    """
    Calculates a dictionary of boolean features (trading conditions) from the data.
    """
    features = {}
    
    # Trend Features (EMA Crossovers)
    ema_fast = ta.ema(df['close'], length=20)
    ema_slow = ta.ema(df['close'], length=50)
    features['ema_20_cross_50'] = ema_fast > ema_slow

    # Momentum Features
    rsi = ta.rsi(df['close'], length=14)
    features['rsi_over_60'] = rsi > 60
    features['rsi_under_40'] = rsi < 40

    # Trend Strength Features
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    features['adx_strong_trend'] = adx['ADX_14'] > 25

    # Volatility Features
    bbands = ta.bbands(df['close'], length=20)
    features['price_above_upper_bb'] = df['close'] > bbands['BBU_20_2.0']
    
    # Combine into a DataFrame
    feature_df = pd.DataFrame(features)
    
    return feature_df


def run_analysis_for_dataset(data_path, results_path):
    os.makedirs(results_path, exist_ok = True)
    print(f"Data path: {data_path}")
    print(f"Results will be saved to {results_path}\n")

    print ("Loading data...")
    df_original = pd.read_csv(data_path, parse_dates = ['timestamp'])
    print(f"Data loaded successfully. Shape {df_original.shape}\n")

    # === STEP 1: Create the new event-based target ===
    df_original['target'] = create_event_target(
        df_original, 
        profit_pct=1.30, 
        loss_pct=0.50, 
        look_forward_bars=30
    )

    # === STEP 2: Calculate all feature conditions ===
    print("Calculating all feature conditions...")
    features_df = calculate_features(df_original)
    
    # Combine original data with features and target
    df_full = pd.concat([df_original, features_df], axis=1)
    df_full.dropna(inplace=True)
    
    print(f"Total events found: {(df_full['target'] == 1).sum()} out of {len(df_full)} samples.")
    if (df_full['target'] == 1).sum() < 10:
        print("!!! WARNING: Very few positive events found. Results may not be reliable.")

    # === STEP 3: Test combinations of features ===
    feature_names = list(features_df.columns)
    all_results = []
    
    # We will test combinations of 2 and 3 features
    for combo_size in range(2, 4): 
        for combo in tqdm(itertools.combinations(feature_names, combo_size), desc=f"Testing combos of {combo_size}"):
            combo_list = list(combo)
            
            X = df_full[combo_list]
            y = df_full['target']

            # Ensure we have both classes in the training set
            if len(np.unique(y)) < 2:
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, shuffle = False, stratify=None # No shuffle for time series
            )
            
            if len(np.unique(y_train)) < 2:
                continue

            # === STEP 4: Use Classification Models ===
            models = {
                'LogisticRegression': LogisticRegression(random_state=42),
                'RandomForestClassifier': RandomForestClassifier(n_estimators=100, random_state=42),
                'XGBClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            }
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if (y_pred == 1).sum() == 0: # Skip if model never predicts a signal
                    continue

                # === STEP 5: Use Classification Metrics ===
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()

                all_results.append({
                    'features': str(combo_list),
                    'model': name,
                    'precision': precision,
                    'recall': recall,
                    'true_positives': tp,
                    'false_positives': fp
                })

    print("\n=== Signal Detection Analysis Complete ===")
    if not all_results:
        print("No valid signals were generated by any model.")
        return

    # Sort results by precision (most important) and number of signals
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by=['precision', 'true_positives'], ascending=[False, False])
    
    results_file = os.path.join(results_path, 'signal_finder_performance.csv')
    results_df.to_csv(results_file, index=False)
    print(f"\nComprehensive signal performance saved to: {results_file}")

    print("\n=== Top 10 Most Precise Signal Combinations ===")
    print(results_df.head(10).to_string(index=False))

def main():
    print("=== Signal Detection Script Started ===\n")
    
    data_paths = [
        # IMPORTANT: Replace this with the actual path to your CSV file.
        '/Users/shawnlevy/Desktop/BootCamp/data/GWPLjamb5ZxrGbTsYNWW7V3p1pAMryZSfaPFTdaEsWgC_1min_ohlcv.csv', 
    ]

    for data_path in data_paths:
        if not os.path.exists(data_path):
            print(f"!!! ERROR: Data file not found at {data_path}")
            print("Please update the 'data_paths' list in the main() function.")
            continue
            
        dataset_name = os.path.splitext(os.path.basename(data_path))[0]
        results_path = f'./results/{dataset_name}_signals'
        
        print(f"\n--- Starting Analysis for {dataset_name} ---")
        run_analysis_for_dataset(data_path, results_path)
        print(f"--- Finished Analysis for {dataset_name} ---\n")


if __name__ == "__main__":
    main()
