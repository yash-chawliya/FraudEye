import os
import joblib
# pyrefly: ignore [missing-import]
import xgboost as xgb
from sklearn.metrics import classification_report, precision_recall_curve, auc

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    print("Warning: imblearn and scikit-learn are incompatible in this environment. Skipping SMOTE.")
    SMOTE_AVAILABLE = False

from preprocess import load_data, preprocess_data, get_train_test_split

def train_model(data_path='../data/creditcard.csv', model_path='../model.pkl', scaler_path='../scaler.pkl'):
    print("Loading data...")
    df = load_data(data_path)
    if df is None:
        print("Please ensure the dataset exists at the specified path.")
        return

    print("Preprocessing data...")
    df = preprocess_data(df, fit_scaler=True, scaler_path=scaler_path)
    
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    
    if SMOTE_AVAILABLE:
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    else:
        print("Skipping SMOTE due to missing dependency.")
        X_train_sm, y_train_sm = X_train, y_train
    
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_sm, y_train_sm)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("Classification Report:\\n", classification_report(y_test, y_pred))
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    print(f"AUC-PR: {pr_auc:.4f}")
    
    print("Saving model...")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    # If run from the src directory, paths should be adjusted
    train_model(data_path='data/creditcard.csv', model_path='model.pkl', scaler_path='scaler.pkl')
