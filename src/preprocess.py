import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df, fit_scaler=True, scaler_path='scaler.pkl'):
    # Feature Engineering
    if 'Time' in df.columns:
        df['Time_Hour'] = (df['Time'] // 3600) % 24

    # Scaling Amount and Time
    features_to_scale = []
    if 'Amount' in df.columns:
        features_to_scale.append('Amount')
    if 'Time' in df.columns:
        features_to_scale.append('Time')
        
    if fit_scaler and features_to_scale:
        scaler = RobustScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        joblib.dump(scaler, scaler_path)
    elif features_to_scale:
        try:
            scaler = joblib.load(scaler_path)
            df[features_to_scale] = scaler.transform(df[features_to_scale])
        except Exception as e:
            print(f"Error loading scaler: {e}")
            
    # Keep the columns as they are, just rename them to indicate they are scaled
    if 'Amount' in df.columns:
        df.rename(columns={'Amount': 'scaled_amount'}, inplace=True)
    if 'Time' in df.columns:
        df.rename(columns={'Time': 'scaled_time'}, inplace=True)

    return df

def get_train_test_split(df):
    if 'Class' not in df.columns:
        raise ValueError("Class column not found in dataframe.")
        
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
