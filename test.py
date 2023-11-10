import unittest
import pickle as pkl
import pandas as pd 
from pydantic import BaseModel
import API
#from fastapi.testclient import TestClient 


#client = TestClient(app) 


# github action 
# requirements.txt 
# liste de toutes les librairie utilisée 
# pip install -r requirements.txt 

# puis heriku pour déploier l'API pour le cloud et récupérer le lien 

# AWS aussi 

# déployer le dashboard streamlit 

# st.dataframe pour les variables et graph 




# charge the model
with open('mlflow_model/best_model.pkl', 'rb') as model_path:
    model = pkl.load(model_path)
X_test = pd.read_csv("X_test.csv")
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

class request_body(BaseModel):
    client_id : int 


class Test_variable(unittest.TestCase):
    def test_prediction(self):
        """
        Test that it can sum a list of integers
        """
        pred = model.predict(X_test.loc[X_test["SK_ID_CURR"] == 160905])
        self.assertEqual(pred, 1)

    def test_probality(self):
        """
        Test that it can sum a list of integers
        """
        proba_pred = model.predict_proba(X_test.loc[X_test["SK_ID_CURR"] == 160905])
        self.assertGreater(proba_pred[0][1], 0.82)

"""        
class Test_API(unittest.TestCase):

    def setUp(self):
        API.app.testing = True
        self.app = API.app.test_client()

    def test_home(self):
        result = self.app.get('/')
        # Make your assertions
"""

if __name__ == '__main__':
    unittest.main()
    
    