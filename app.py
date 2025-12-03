from flask import Flask, request, jsonify
from flask_cors import CORS   # <- import CORS
import joblib
import pandas as pd

# Initialiser Flask
app = Flask(__name__)
CORS(app)

# Chemins vers les artifacts
ARTIFACTS_DIR = "./artifacts"
MODEL_FILE = f"{ARTIFACTS_DIR}/gb_model.joblib"
SCALER_FILE = f"{ARTIFACTS_DIR}/scaler.joblib"
FEATURES_FILE = f"{ARTIFACTS_DIR}/feature_list.joblib"

# Charger le modèle, scaler et liste de features
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
feature_list = joblib.load(FEATURES_FILE)

# Fonction pour créer les features comme à l'entraînement
def create_features(df):
    # s'assurer que le max BMI est > 30 pour éviter les bords dupliqués
    max_bmi = max(df['BMI'].max(), 30.1)
    bins = [0, 18.5, 25, 30, max_bmi]
    labels = ['Underweight', 'Normal_Weight', 'Overweight', 'Obese']
    
    # créer la colonne BMI_Category sans erreur si bords dupliqués
    df['BMI_Category'] = pd.cut(
        df['BMI'],
        bins=bins,
        labels=labels,
        right=False,
        duplicates='drop'
    )
    
    df['HighRisk_Cardio'] = ((df['HighBP'] == 1.0) & (df['HighChol'] == 1.0)).astype(int)
    df['BMI_PhysActivity_Interaction'] = df['BMI'] * (1 - df['PhysActivity'])
    return df

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer le JSON
        data = request.get_json()
        df = pd.DataFrame(data)

        # Créer les features
        df = create_features(df)

        # Préparer X
        X = df[feature_list].copy()
        numeric_cols = [
            c for c in ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Income', 'BMI_PhysActivity_Interaction']
            if c in X.columns
        ]
        X[numeric_cols] = scaler.transform(X[numeric_cols])

        # Prédictions
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # Retourner le résultat
        results = [{"prediction": int(pred), "probability": float(prob)} for pred, prob in zip(y_pred, y_proba)]
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Lancer le serveur
if __name__ == "__main__":
    app.run(debug=True)
