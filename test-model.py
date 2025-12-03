import joblib
import pandas as pd

# Artifacts
ARTIFACTS_DIR = "./artifacts"
MODEL_FILE = f"{ARTIFACTS_DIR}/gb_model.joblib"
SCALER_FILE = f"{ARTIFACTS_DIR}/scaler.joblib"
FEATURES_FILE = f"{ARTIFACTS_DIR}/feature_list.joblib"

# Charger modèle, scaler, features
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
feature_list = joblib.load(FEATURES_FILE)

# Ton exemple
df = pd.DataFrame([{
    "BMI": 28.0,
    "PhysActivity": 1,
    "HighBP": 0,
    "HighChol": 1,
    "GenHlth": 3,
    "MentHlth": 1,
    "PhysHlth": 2,
    "Age": 45,
    "Income": 4,
    "DiffWalk": 0,
    "HeartDiseaseorAttack": 0,
    "Stroke": 0,
    "HvyAlcoholConsump": 0
}])

# Fonction create_features
def create_features(df):
    df['HighRisk_Cardio'] = ((df['HighBP'] == 1) & (df['HighChol'] == 1)).astype(int)
    df['BMI_PhysActivity_Interaction'] = df['BMI'] * (1 - df['PhysActivity'])
    return df

df = create_features(df)

# Préparer X
X = df[feature_list].copy()

# Colonnes numériques à scaler
numeric_cols = [c for c in ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Income', 'BMI_PhysActivity_Interaction'] if c in X.columns]
X[numeric_cols] = scaler.transform(X[numeric_cols])

# Prédire
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

# Afficher
for i, (pred, proba) in enumerate(zip(y_pred, y_proba)):
    print(f"Ligne {i}: Prédit={pred}, Probabilité={proba:.3f}")
