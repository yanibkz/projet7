import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import numpy as np
from pandas import json_normalize
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import json
# Fonctions pour tracer les graphiques
def plot_loan_percentage(df):
    st.write("### Pourcentage de prêts remboursés")
    fig = plt.figure()
    plt.pie(df['TARGET'].value_counts() / len(df) * 100,
            autopct='%1.1f%%',
            labels=["Remboursé sans aucun problème", "Un incident de paiement est survenu"]
            )
    plt.title('Pourcentage de prêts remboursés')
    st.pyplot(fig)


def plot_age_distribution(df):
    st.write("### Distribution de l'âge du client")
    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 5))
    plt.hist(df['DAYS_BIRTH'] / 365, edgecolor='k', bins=25)
    plt.title('Age du Client')
    plt.xlabel('Age (années)')
    plt.ylabel('Count')
    st.pyplot(plt.gcf())


# Modifiez la signature de la fonction pour accepter un DataFrame
def plot_feature_importance(df):
    st.write("### Importance des features")

    # Faites une requête à votre API pour obtenir les coefficients
    response = requests.get('https://projet7op-954d4e3f556b.herokuapp.com/features')
    if response.status_code == 200:
        data = response.json()
        df_coef = json_normalize(data)  # Convertir le JSON en DataFrame
        df_coef = df_coef.sort_values('Coefficient', ascending=False)  # Trier par importance

        # Créez le graphique
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_coef, x='Coefficient', y='Feature', palette='viridis')
        plt.title('Importance des fonctionnalités')
        plt.xlabel('Coefficient')
        plt.ylabel('Fonctionnalité')
        st.pyplot(plt.gcf())
    else:
        st.write("Erreur lors de la récupération de l'importance des fonctionnalités.")



def information_client_page(df):
    st.title("Information Client")

    # Convertit DAYS_BIRTH en âge en années
    df['AGE_YEARS'] = -(df['DAYS_BIRTH'] / 365.25)

    # Ajout de sliders pour la sélection de la plage d'âge
    min_age = st.slider("Âge minimum", min_value=19, max_value=65, value=19)
    max_age = st.slider("Âge maximum", min_value=19, max_value=65, value=65)

    # Filtrage du dataframe selon la plage d'âge
    filtered_df = df[(df['AGE_YEARS'] >= min_age) & (df['AGE_YEARS'] <= max_age)]

    # Affiche le DataFrame filtré
    st.dataframe(filtered_df, width=1000, height=400)

    # Affiche le nombre total de clients dans le DataFrame filtré
    st.write(f"### Nombre total de clients dans la plage d'âge sélectionnée: {filtered_df.shape[0]}")

    # Mise à jour des graphiques et statistiques pour refléter le DataFrame filtré
    plot_loan_percentage(filtered_df)
    plot_age_distribution(filtered_df)
    plot_feature_importance(filtered_df)




def plot_gauge(probability, threshold=0.58):
    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=probability,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 1], "tickvals": [0, threshold, 1], "ticktext": ["Refusé", "Limite", "Accepté"]},
            "bar": {"color": "black"},
            "steps": [
                {"range": [0, threshold], "color": "red"},
                {"range": [threshold, 1], "color": "green"}
            ],
            "threshold": {"line": {"color": "blue", "width": 4}, "thickness": 0.75, "value": threshold}
        }
    ))

    st.plotly_chart(fig)

# Fonction credit_prediction_page
def credit_prediction_page(df):
    if 'sk_id' not in st.session_state:
        st.session_state.sk_id = None

    st.title("Prédiction de Crédit")
    client_id = st.text_input("Entrez l'ID du client :")

    if st.button("Effectuer la prédiction"):
        try:
            st.session_state.sk_id = int(client_id)
            prediction_response = requests.get(f'https://projet7op-954d4e3f556b.herokuapp.com/predict/{st.session_state.sk_id}')

            if prediction_response.status_code == 200:
                prediction_data = prediction_response.json()
                prediction_df = pd.DataFrame.from_dict(prediction_data)

                prediction = prediction_df['Prediction'][0]
                probability = prediction_df['Probability'][0]

                # Afficher la jauge de probabilité
                plot_gauge(probability)

                st.write(f"ID du Client : {st.session_state.sk_id}")
                if prediction == 1:
                    st.markdown("<span style='color:green'>Le crédit est accordé.</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span style='color:red'>Le crédit n'est pas accordé.</span>", unsafe_allow_html=True)

            else:
                st.write(f"Erreur lors de la récupération de la prédiction. Code: {prediction_response.status_code}")

        except ValueError:
            st.write("Veuillez entrer un ID de client valide (nombre entier).")


    if st.session_state.sk_id is not None:
        column_to_display = st.selectbox("Choisissez une colonne pour afficher sa distribution :", df.columns.tolist())
        selected_value = df[df['SK_ID_CURR'] == st.session_state.sk_id][column_to_display].values[
            0]  # Prend la valeur du client sélectionné pour la colonne sélectionnée

        if st.button("Afficher la distribution"):
            try:
                # histogramme
                fig = px.histogram(df, x=column_to_display, color="TARGET",
                                   title=f"Distribution de {column_to_display} par statut de crédit",
                                   labels={"TARGET": "Statut du Crédit"})

                # Ajout d'un marqueur pour le client sélectionné
                fig.add_scatter(x=[selected_value], y=[0], mode='markers', marker=dict(size=16, color='red'),
                                name='Client sélectionné')

                st.plotly_chart(fig)

            except Exception as e:
                st.write(str(e))

    if st.button("Obtenir le graphique SHAP"):
        if st.session_state.sk_id is not None:  # Utilisation de l'ID client de la session
            response = requests.get(f"https://projet7op-954d4e3f556b.herokuapp.com/shap_plot/{st.session_state.sk_id}")

            if response.status_code == 200:
                st.image(response.content, caption="Graphique SHAP", use_column_width=True)
            else:
                st.write(f"Erreur lors de la récupération du graphique SHAP. Code: {response.status_code}")


def get_prediction(sk_id):
    response = requests.get('https://projet7op-954d4e3f556b.herokuapp.com/{sk_id}')
    print("API Response:", response.json())  # Ajout d'un print pour déboguer
    if response.status_code == 200:
        return response.json()[0]['Prediction']
    else:
        return None


# Votre liste de colonnes sélectionnées
selected_columns = [
    'ACTIVE_MONTHS_BALANCE_SIZE_MEAN',
    'BURO_CREDIT_ACTIVE_Closed_MEAN',
    'BURO_DAYS_CREDIT_MEAN',
    'BURO_MONTHS_BALANCE_SIZE_MEAN',
    'CC_AMT_BALANCE_MEAN',
    'CC_AMT_RECEIVABLE_PRINCIPAL_MEAN',
    'CC_AMT_RECIVABLE_MEAN',
    'CC_AMT_TOTAL_RECEIVABLE_MEAN',
    'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN',
    'CC_CNT_DRAWINGS_CURRENT_MAX',
    'CC_CNT_DRAWINGS_CURRENT_MEAN',
    'CC_COUNT',
    'DAYS_BIRTH',
    'DAYS_EMPLOYED_PERC',
    'EXT_SOURCE_1',
    'EXT_SOURCE_2',
    'EXT_SOURCE_3',
    'PREV_CODE_REJECT_REASON_XAP_MEAN',
    'PREV_NAME_CONTRACT_STATUS_Approved_MEAN',
    'SK_ID_CURR'
]

# Code principal de l'application Streamlit
if __name__ == '__main__':


    # Authentification au milieu de la page
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.header("Connexion")
        user = st.text_input("Identifiant")
        password = st.text_input("Mot de passe", type="password")

    if user == "yani" and password == "azerty":
        initial_df = pd.read_csv('df_final.csv')

        app_mode = st.sidebar.selectbox(
            "Sélectionnez un onglet :", ["Information Client", "Prédiction de Crédit"])

        uploaded_file = st.sidebar.file_uploader(
            "Choisissez un fichier CSV", type=["csv"])

        if uploaded_file is not None:
            with open("temp_file.csv", "wb") as f:
                f.write(uploaded_file.read())

            response = requests.post(
                'https://projet7op-954d4e3f556b.herokuapp.com/import',
                files={'file': open("temp_file.csv", "rb")}
            )

            if response.status_code == 200:
                try:
                    print("API Response:", response.json())
                except json.JSONDecodeError:
                    print("La réponse n'est pas au format JSON:", response.text)
            else:
                print("Erreur API:", response.status_code)

            df = pd.read_csv("temp_file.csv")

        else:
            df = initial_df

        if app_mode == "Information Client":
            information_client_page(df)
        elif app_mode == "Prédiction de Crédit":
            credit_prediction_page(df)
    else:
        st.write("Veuillez vous authentifier pour accéder aux fonctionnalités.")