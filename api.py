from flask import Flask, jsonify, request, send_file
import os
import pandas as pd
from joblib import load
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
import seaborn as sns


import io
app = Flask(__name__)

# Charger les ressources au démarrage de l'application
model = load('best_lr_model.joblib')
initial_df = pd.read_csv('df_final.csv')
initial_df.drop(['TARGET'], axis=1, inplace=True)
scaler = StandardScaler()
scaled_initial_data = scaler.fit_transform(initial_df.drop('SK_ID_CURR', axis=1))

# Initialisez le calculateur SHAP avec l'ensemble de données complet
explainer = shap.Explainer(model.named_steps['estimator'], scaled_initial_data)

best_threshold = 0.6873417721518987
current_df = initial_df


@app.route('/import', methods=['POST'])
def import_df():
    global current_df, scaler

    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        # Mettre à jour le DataFrame
        current_df = pd.read_csv(uploaded_file)

        # Mettre à jour le scaler
        scaled_data = scaler.fit_transform(current_df.drop('SK_ID_CURR', axis=1))

        return jsonify({"message": "DataFrame and scaler updated successfully"})
    else:
        return jsonify({"error": "No file uploaded"}), 400

@app.route('/predict/<int:sk_id>', methods=['GET'])
def predict(sk_id):
    global current_df
    filtered_df = current_df[current_df['SK_ID_CURR'] == sk_id]

    if filtered_df.empty:
        return jsonify({"error": "SK_ID_CURR not found"}), 404

    scaled_data = scaler.transform(filtered_df.drop('SK_ID_CURR', axis=1))

    # Effectuer la prédiction et récupérer les probabilités
    prediction_proba = model.predict_proba(scaled_data)[:, 1][0]
    best_threshold = 0.58
    prediction = (prediction_proba > best_threshold).astype(int)

    # Conversion en DataFrame et ajout de la probabilité
    prediction_df = pd.DataFrame({'SK_ID_CURR': [sk_id], 'Prediction': prediction, 'Probability': prediction_proba})

    return jsonify(prediction_df.to_dict(orient='records'))

@app.route('/features', methods=['GET'])
def features():
    try:
        # Obtenir les coefficients à partir de l'étape de régression logistique dans le pipeline
        coefficients = model.named_steps['estimator'].coef_.tolist()[0]
    except KeyError:
        return jsonify({"error": "Logistic Regression step not found in pipeline"}), 404

    # Mettre les noms de colonnes et les coefficients dans un DataFrame
    coef_df = pd.DataFrame({
        'Feature': initial_df.drop('SK_ID_CURR', axis=1).columns.tolist(),
        'Coefficient': coefficients
    })

    return jsonify(coef_df.to_dict(orient='records'))
from flask import send_file

@app.route('/features_plot', methods=['GET'])
def features_plot():
    try:
        coefficients = model.named_steps['estimator'].coef_.tolist()[0]
    except KeyError:
        return jsonify({"error": "Logistic Regression step not found in pipeline"}), 404

    coef_df = pd.DataFrame({
        'Feature': initial_df.drop('SK_ID_CURR', axis=1).columns.tolist(),
        'Coefficient': coefficients
    })

    plt.figure(figsize=(10, 6))
    coef_df = coef_df.sort_values(by='Coefficient', ascending=False)
    plt.barh(coef_df['Feature'], coef_df['Coefficient'])
    plt.xlabel('Coefficient')
    plt.ylabel('Feature')
    plt.title('Feature Importance')

    plt.savefig('feature_importance.png')

    return send_file('feature_importance.png', mimetype='image/png')

# Initialisez le calculateur SHAP une fois.
explainer = shap.Explainer(model.named_steps['estimator'], scaled_initial_data[:100])










@app.route('/distribution-plots/<int:sk_id>/<string:column>', methods=['GET'])
def distribution_plots(sk_id, column):
    if column not in initial_df.columns:
        return jsonify({"error": "Invalid column name"}), 400

    fig, ax = plt.subplots(figsize=(8, 4))
    client_data = initial_df[initial_df['SK_ID_CURR'] == sk_id]

    sns.kdeplot(data=initial_df, x=column, ax=ax)
    if not client_data.empty:
        client_value = client_data.iloc[0][column]
        ax.axvline(client_value, color='r', linestyle='--')
    ax.set_title(f'Distribution de {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Densité')

    plt.tight_layout()
    plt.savefig('distribution_single_plot.png')
    plt.close()

    return send_file('distribution_single_plot.png', mimetype='image/png')


@app.route('/', methods=['GET'])
def home():
    return "Bienvenue sur l'API de prediction."


if __name__ == '__main__':
    # Spécifiez le numéro de port
    port = int(os.environ.get('PORT', 5003))

    # Lancez l'application en écoutant sur le port spécifié
    app.run(host='0.0.0.0', port=port)
