import pandas as pd
import pickle
from sqlalchemy import create_engine
import psycopg2 
import io


def add_missing_dummy_columns( d, columns):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0

def fix_columns( d, columns ):
    add_missing_dummy_columns( d, columns )
    assert( set( columns ) - set( d.columns ) == set())
    extra_cols = set( d.columns ) - set( columns )
    d = d[ columns ]
    return d

class PredictFraud():
    
    def __init__(self, df):
        self.df = df
        self.newdf = df
        self.export_df = None
        self.new_data = None
        
    def manipulate_data(self):
        with open('high_risk_list', 'rb') as f:
            high_risk = pickle.load(f)
        with open('columns', 'rb') as c:
            columns = pickle.load(c)
        self.df['high_risk_loc'] = [1 if x in high_risk else 0 for x in self.df['venue_state']]
        self.df['user_type'].astype('object', copy = False)
        self.df = pd.get_dummies(self.df, columns = ['currency', 'user_type'], drop_first = True)
        self.df['listed'] = [1 if x == 'y' else 0 for x in self.df['listed']]
        self.df['event_duration'] = self.df['event_end'] - self.df['event_start']
        self.df.drop(['country', 'venue_address', 'venue_country', 'venue_latitude', 'venue_longitude', 'venue_state', 'event_created', 'event_end', 'event_published', 'event_start', 'user_created'], axis = 1, inplace = True)
        self.df.drop(list(self.df.select_dtypes(include=['object']).columns), axis = 1, inplace = True)
        self.df.fillna(0, inplace = True)
        self.df = fix_columns(self.df, columns)
        
    
    def predict_fraud(self):
        with open('rf_model', 'rb') as m:
            rf_model = pickle.load(m)
        predictions = rf_model.predict_proba(self.df)
        self.df['prediction'] = predictions[:,1].astype(int)

    def export_to_db(self):
        self.export_df = pd.concat((self.df['object_id'],self.df['prediction']), axis = 1)
        #engine = create_engine('postgresql+psycopg2://chrisjoetrevjason:ElephantBacon@frauddb.cz8soh9v5z3q.us-west-1.rds.amazonaws.com:5432/frauddbone')
        engine = create_engine('postgresql+psycopg2://josephdoperalski:@localhost:5432/predictions')
        conn = engine.raw_connection()
        cur = conn.cursor()
        output = io.StringIO()
        self.export_df.to_csv(output, sep='\t', header=False, index=False)
        output.seek(0)
        contents = output.getvalue()
        cur.copy_from(output, 'predictions', null="") # null values become ''
        conn.commit()
        
        