import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import os




def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data', 'train.csv')
    return pd.read_csv(data_path)

def get_preprocessor():
    cols_to_scale = ['Battery Power','Microprocessor Speed','Front Camera','Internal Memory','Mobile Depth','Mobile Weight',
                     'Number of Cores','Primary Camera','Pixel Resolution Height','Pixel Resolution Width','RAM','Screen Height','Screen Width','Talk Time']
    return ColumnTransformer([
        ('scaler', StandardScaler(), cols_to_scale)
    ], remainder='passthrough')
