# train_model.py
"""
Script d'entraînement pour le modèle de prédiction du diabète.
Usage:
    python train_model.py --csv path/to/diabetes.csv --out_dir ./artifacts
"""

import argparse
import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------- Fonctions utilitaires ----------------------------

def load_data(path):
    df = pd.read_csv(path)
    return df

def drop_duplicates(df):
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def cap_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

def create_features(df):
    # Discretisation BMI (on conserve la variable numeric originale aussi)
    bins = [0, 18.5, 25, 30, df['BMI'].max()]
    labels = ['Underweight', 'Normal_Weight', 'Overweight', 'Obese']
    df['BMI_Category'] = pd.cut(df['BMI'], bins=bins, labels=labels, right=False)

    # High risk cardio
    df['HighRisk_Cardio'] = np.where((df['HighBP'] == 1.0) & (df['HighChol'] == 1.0), 1, 0)

    # Interaction BMI * (1 - PhysActivity)
    df['BMI_PhysActivity_Interaction'] = df['BMI'] * (1 - df['PhysActivity'])

    return df

def select_features_chi2(df, target_col='Diabetes_binary', k=15):
    X = df.drop([target_col, 'BMI_Category'], axis=1, errors='ignore')
    y = df[target_col]

    # Chi2 needs non-negative integers -> cast
    X_int = X.fillna(0).astype(int)
    y_int = y.astype(int)

    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X_int, y_int)

    scores = pd.DataFrame({
        'feature': X.columns,
        'chi2_score': selector.scores_
    }).sort_values(by='chi2_score', ascending=False)

    top_k = scores.head(k)['feature'].tolist()
    return top_k, scores

def prepare_X_y(df, feature_list, target_col='Diabetes_binary'):
    X = df[feature_list].copy()
    y = df[target_col].copy()
    return X, y

def scale_numeric_columns(X_train, X_test, numeric_cols):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    scaler.fit(X_train[numeric_cols])
    X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train_scaled, X_test_scaled, scaler

def train_and_evaluate(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(random_state=42)
    start = time.time()
    model.fit(X_train, y_train)
    duration = time.time() - start
    print(f"Training completed in {duration:.2f}s")

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    return model

def cross_val_score_model(model, X, y, cv_splits=5):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    # ⚠️ Windows + OneDrive workaround: n_jobs=1 pour éviter loky errors
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=1)
    
    print(f"Cross-val F1 scores: {scores}")
    print(f"Mean F1: {scores.mean():.4f}, Std: {scores.std():.4f}")

def save_artifacts(out_dir, model, scaler, feature_list):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, 'gb_model.joblib'))
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))
    joblib.dump(feature_list, os.path.join(out_dir, 'feature_list.joblib'))
    print(f"Saved model, scaler, feature_list in {out_dir}")

# ---------------------------- Main ----------------------------

def main(args):
    print("Loading data...")
    df = load_data(args.csv)

    print("Dropping duplicates...")
    df = drop_duplicates(df)

    print("Capping outliers (IQR) on BMI, MentHlth, PhysHlth ...")
    df = cap_outliers_iqr(df, cols=['BMI', 'MentHlth', 'PhysHlth'])

    print("Creating features...")
    df = create_features(df)

    print("Selecting top features via Chi2...")
    top_k, scores_df = select_features_chi2(df, target_col='Diabetes_binary', k=15)
    print("Top features selected:")
    for f in top_k:
        print(" -", f)

    # ensure interaction feature is included
    if 'BMI_PhysActivity_Interaction' not in top_k:
        top_k.append('BMI_PhysActivity_Interaction')

    feature_list = top_k

    print("Preparing X and y...")
    X, y = prepare_X_y(df, feature_list, target_col='Diabetes_binary')

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_cols = [c for c in ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income', 'BMI_PhysActivity_Interaction'] if c in X_train.columns]
    print("Numeric columns to scale:", numeric_cols)

    X_train_scaled, X_test_scaled, scaler = scale_numeric_columns(X_train, X_test, numeric_cols)

    print("Performing cross-validation of GradientBoosting (F1)...")
    cross_val_score_model(GradientBoostingClassifier(random_state=42), X_train_scaled, y_train, cv_splits=5)

    print("Training final model on scaled data...")
    model = train_and_evaluate(X_train_scaled, y_train, X_test_scaled, y_test)

    print("Saving artifacts...")
    save_artifacts(args.out_dir, model, scaler, feature_list)
    print("Done.")

# ---------------------------- CLI ----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    parser.add_argument("--out_dir", default="./artifacts", help="Where to save model/scaler/features")
    args = parser.parse_args()
    main(args)
