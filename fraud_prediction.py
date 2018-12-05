import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, roc_curve
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import pickle


class PredictFraud():
    
    def __init__(self, df):
        self.df = df
        
    def manipulate_data(self):
        with open('high_risk_list', 'rb') as f:
            high_risk = pickle.load(f)
        self.df['high_risk_loc'] = [1 if x in high_risk else 0 for x in self.df['venue_state']]
        self.df['user_type'].astype('object', copy = False)
        self.df = pd.get_dummies(self.df, columns = ['currency', 'user_type'], drop_first = True)
        self.df['listed'] = [1 if x == 'y' else 0 for x in self.df['listed']]
        self.df['event_duration'] = self.df['event_end'] - self.df['event_start']
        self.df.drop(['country', 'venue_address', 'venue_country', 'venue_latitude', 'venue_longitude', 'venue_state', 'acct_type', 'approx_payout_date', 'event_created', 'event_end', 'event_published', 'event_start', 'user_created', 'sale_duration2'], axis = 1, inplace = True)
        self.df.drop(list(self.df.select_dtypes(include=['object']).columns), axis = 1, inplace = True)
        self.df.fillna(0, inplace = True)
    
    def predict_fraud(self):
        with open('rf_model', 'rb') as m:
            rf_model = pickle.load(m)
        predictions = rf_model.predict_proba(self.df)
        self.df['prediction'] = [1 if p > .2 else 0 for p in predictions[:,1]]
        