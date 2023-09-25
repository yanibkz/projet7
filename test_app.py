import json
import pytest
import pandas as pd

from api import app  # Remplacez 'api' par le nom du fichier où se trouve votre application Flask
import random

# Prendre un ID au hasard depuis le DataFrame initial_df

initial_df = pd.read_csv('df_final.csv')
random_id = random.choice(initial_df['SK_ID_CURR'].tolist())

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict(client):
    response = client.get(f'/predict/{random_id}')  # Utilisez un ID qui est effectivement dans initial_df
    assert response.status_code == 200
    # Vérifiez le contenu de la réponse ici
def test_features(client):
    response = client.get('/features')
    assert response.status_code == 200
    # Vérifiez le contenu de la réponse ici
def test_features_plot(client):
    response = client.get('/features_plot')
    assert response.status_code == 200
    assert response.mimetype == 'image/png'

def test_shap_plot(client):
    response = client.get(f'/shap_plot/{random_id}')  # Utilisez un ID qui est effectivement dans initial_df
    assert response.status_code == 200
    assert response.mimetype == 'image/png'

def test_distribution_plots(client):
    response = client.get('/distribution-plots/1/SK_ID_CURR')  # Utilisez un ID et une colonne qui sont effectivement dans initial_df
    assert response.status_code == 200
    assert response.mimetype == 'image/png'