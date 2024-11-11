import pandas as pd
import numpy as np

def preprocess_data(data, is_banknote=False):
    if is_banknote:
        # Preprocess the banknote dataset
        ground_truth = data['class']
        data = data.drop(columns=['class'])
    
        # Normalize the data
        data_normalized = (data - data.mean()) / data.std()
        
        # Convert ground truth to numpy array
        Y = ground_truth.to_numpy()
        
        # Convert data to numpy array
        X = data_normalized.to_numpy()
        
        # Combine X and Y into a single dataset
        data_processed = np.column_stack((X, Y))
        
    else:
        # Preprocess the kidney disease dataset
        data.columns = data.columns.str.strip()
        data.replace('?', np.nan, inplace=True)
        data = data.apply(pd.to_numeric, errors='ignore')
        
        categorical_features = ['sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        for feature in categorical_features:
            data[feature] = data[feature].astype('category')
        
        ground_truth = data['classification']
        data = data.drop(columns=['classification'])
        
        numeric_features = data.select_dtypes(exclude='category').columns
        for feature in numeric_features:
            data[feature] = pd.to_numeric(data[feature], errors='coerce')
            data[feature].fillna(data[feature].mean(), inplace=True)
        
        for feature in categorical_features:
            most_frequent_value = data[feature].mode().iloc[0]
            data[feature].fillna(most_frequent_value, inplace=True)
        
        data_encoded = pd.get_dummies(data, columns=categorical_features)
        
        boolean_columns = data_encoded.select_dtypes(include=['bool']).columns
        data_encoded[boolean_columns] = data_encoded[boolean_columns].astype('int')
        
        data_normalized = (data_encoded - data_encoded.mean()) / data_encoded.std()
        
        ground_truth_numeric = ground_truth.map(lambda x: 1 if x == 'ckd' else 0).to_numpy()
        
        Y = ground_truth_numeric
        X = data_normalized.to_numpy()
        
        data_processed = np.column_stack((X, Y))
    
    return data_processed

# Example usage:
data_kidney = pd.read_csv('data/kidney_disease.csv')
data_banknote = pd.read_csv('data/data_banknote_authentication.txt', names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])

processed_kidney = preprocess_data(data_kidney)
processed_banknote = preprocess_data(data_banknote, is_banknote=True)