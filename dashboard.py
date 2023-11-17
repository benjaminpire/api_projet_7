import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import requests
import json
import seaborn as sns
import lime.lime_tabular
import re
import plotly.graph_objects as go
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go


#### objectifs :

# 1.Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.

# 2.Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).

# 3.Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

# diagramme en jauge de chez plotly 



########
#Fonction de prediction 
########
def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'client_id': data}

    response = requests.post(headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

########
#Fonction principale
########
def main():
    
    # Get the data frames
    df = pd.read_csv( 'X_test.csv')
    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv')
    
    # Link to the url
    MLFLOW_URI = 'https://newprojet7-b66a612de84a.herokuapp.com/predict'
    
    # Initialisation des pages
    tab1, tab2, tab3 = st.tabs(["Informations client", "Comparaison avec d'autres clients", "Score interprétation"])

    ### Premiere page
    with tab1:
        # selection of the client 
        client_select = st.selectbox(
                "Select **the ID** of the client",
               df["SK_ID_CURR"])
        
        col_list = ["SK_ID_CURR", "DAYS_BIRTH","CNT_CHILDREN", "FLAG_EMP_PHONE", "FLAG_EMAIL", "NAME_EDUCATION_TYPE_Highereducation","CODE_GENDER", "AMT_CREDIT", "AMT_INCOME_TOTAL"]
        st.dataframe(
                df.loc[df["SK_ID_CURR"] == client_select, col_list],
                column_config={
                    "SK_ID_CURR": "ID du client",
                    "DAYS_BIRTH": "Années de naissance",
                    "CNT_CHILDREN": "numbers of children",
                    "FLAG_EMP_PHONE": "Personal phone",
                    "FLAG_EMAIL": "Mail",
                    "NAME_EDUCATION_TYPE_Highereducation": "Higher education",
                    "CODE_GENDER": "genre",
                    "AMT_CREDIT": "Montant du crédit",
                    "AMT_INCOME_TOTAL": "Total des entrans"},
                height = 75)
        

        predict_btn = st.button('Prédire')

        col1, col2 = st.columns(2)
        if predict_btn:
            pred = None
            pred = request_prediction(MLFLOW_URI,client_select)
            st.metric("Le client selectioné  est", value=pred["class"])
        
    ### Deuxieme page
    with tab2:
        variable = st.multiselect('Selectionnez quelques variables',
                                  df.columns) 
        col_binaire = df.loc[:,[col for col in df.columns if 'FLAG' in col]].columns.tolist()
        col_binaire.append('CODE_GENDER')
        col1, col2 = st.columns([1, 4])
        for i in range(len(variable)):
            df["str_SK_ID_CURR"] = df["SK_ID_CURR"].astype(str)
            client_value = df.loc[df["SK_ID_CURR"] == client_select, variable[i]]
            if variable[i] in col_binaire: 
                with col1:
                    st.metric(label=variable[i],
                              value=client_value)
            else: 
                with col2:
                    colors = {client_select: "red"}
                    color_discrete_map = {c: colors.get(c, "blue") for c in df["SK_ID_CURR"]}
                    fig = px.strip(df, x = variable[i])
                    fig2 = px.strip(df.loc[df["SK_ID_CURR"] == client_select], x = variable[i], color_discrete_sequence = ["red"]).data
                    fig.add_traces(fig2)
                    st.plotly_chart(fig)

        
    ### troisième page
    with tab3:
        
        # Initialisation des colonnes
        col1, col2 = st.columns([1,2])

        # selectionner quelques variables
        if predict_btn:
            
            # Affichages de la probabilité
            with col1:
                pred = None
                pred = request_prediction(MLFLOW_URI,client_select)
                st.metric("Le client selectioné  est", value=pred["class"])
            
            with col2:
                st.metric("Avec une probabilité de", value=pred["proba"])
                #jauge = go.Figure(go.Indicator(mode = "number+gauge",gauge = {'shape': "bullet"},delta = {'reference': 1},value = pred["proba"],domain = {'x': [0, 0.5], 'y': [0, 0.5]},title = {'text': "probabilité"}))

                #st.plotly_chart(jauge)

                
            # Préparation du dataframe de l'interprétabilité locale
            list_inter = pred["inter"]
            feature_names = [explanation[0] for explanation in list_inter]
            contributions = [float(explanation[1]) for explanation in list_inter]
            df_inter = pd.DataFrame({'Feature Names': feature_names, 'Contributions': contributions})
            df_inter['autorisation/refus'] = df_inter['Contributions'].apply(lambda x: 'autorisation' if x < 0 else 'refus')
            df_inter_sorted = df_inter.sort_values(by=['Contributions'], key=abs)
            
            # Affichages de l'interprétabilité locale
            fig_bar = px.bar(df_inter_sorted,
                             x="Contributions",
                             y="Feature Names", 
                             color=df_inter_sorted['autorisation/refus'])
            st.plotly_chart(fig_bar)
            
            

########
#Appel de la fonction principale
########
if __name__ == '__main__':
    main()
