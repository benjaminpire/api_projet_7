# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import joblib
import pandas as pd 
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
import lime.lime_tabular

# Declaring our FastAPI instance
app = FastAPI()

X_test = pd.read_csv("X_test.csv")
X_train = pd.read_csv("X_train.csv")

class request_body(BaseModel):
    client_id : int 

# Defining path operation for root endpoint
@app.get('/')
def main():
    print("hello")

@app.post('/predict')
def predict(data : request_body):

    # charge the model
    with open('mlflow_model/best_model.joblib', 'rb') as model_path:
        model = joblib.load(model_path)
    
    #récupération de l'interprétabilité :
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=X_train.columns,
        class_names=['mauvais client', 'bon client'],
        mode='classification'
    )
    print(model)
    exp = explainer.explain_instance(
        data_row=np.array(X_test.loc[X_test["SK_ID_CURR"] == data.client_id])[0], 
        predict_fn=model.predict_proba)
    # Extract relevant information from exp for JSON response
    explanation_data = {
        "as_list": exp.as_list(),
        "show_table": exp.show_in_notebook(show_table=True)  
    }

    exp.show_in_notebook(show_table=True)
    
    
    # Predicting the Class
    pred = model.predict(X_test.loc[X_test["SK_ID_CURR"] == data.client_id])
    print(pred.tolist())
    print(pred.tolist())
    proba_pred = model.predict_proba(X_test.loc[X_test["SK_ID_CURR"] == data.client_id])
    # Return the Result
    print(pred)
    print(proba_pred)
    proba = None 
    valid_pret = None
    if pred[0]==0: 
        valid_pret = 'refusé'
        proba = proba_pred[0][0]
    else :
        valid_pret = 'autorisé'
        proba = proba_pred[0][1]
    print("this is it" + str(exp))
    return {'class' : valid_pret, 'proba' : proba, 'inter' : exp.as_list()}

#'inter' : exp