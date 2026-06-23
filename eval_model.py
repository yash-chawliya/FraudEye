import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
)
from src.preprocess import preprocess_data

def evaluate():
    # Load dataset
    print("Loading data...")
    df = pd.read_csv('test_dataset.csv')
    
    # Preprocess
    print("Preprocessing data...")
    y_true = df['Class']
    X = preprocess_data(df.copy(), fit_scaler=False, scaler_path='scaler.pkl')
    if 'Class' in X.columns:
        X = X.drop(columns=['Class'])
        
    # Load model
    print("Loading model...")
    model = joblib.load('model.pkl')
    
    # Predict
    print("Predicting...")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    # Calculate metrics
    print("Calculating metrics...")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    metrics = {
        'Accuracy': float(acc),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1-Score': float(f1),
        'ROC-AUC': float(roc_auc)
    }
    print("METRICS_JSON_START")
    print(json.dumps(metrics))
    print("METRICS_JSON_END")
    
    # Plotting styles
    sns.set_theme(style="whitegrid")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legitimate', 'Fraud'], yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('assets/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('assets/roc_curve.png', dpi=300)
    plt.close()
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})', color='green', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig('assets/pr_curve.png', dpi=300)
    plt.close()
    
    print("Saved plots to assets/")

if __name__ == '__main__':
    evaluate()
