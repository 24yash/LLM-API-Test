from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import numpy as np
from flask_cors import CORS 


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'abcd'


def apply_preprocessing(options, target):
    # data = pd.read_csv('uploads/' + filename)
    # test = pd.read_csv('uploads/' + test_file_path)  
    data = pd.read_csv('try/train.csv')
    test = pd.read_csv('try/test.csv')
    target_column = target 
    exclude_columns = ['id', 'ID', 'Id']


    # Ensure the target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in the dataset.")
    
    # Handle null values for numerical columns
    numerical_columns = data.select_dtypes(include=np.number).columns.tolist()
    numerical_columns = [col for col in numerical_columns if col not in exclude_columns]  # Exclude specified columns
    if target_column in numerical_columns:
        numerical_columns.remove(target_column)  # Exclude the target column from preprocessing
    
    if options['null_handling'] == 'drop':
        data.dropna(subset=numerical_columns, inplace=True)
    elif options['null_handling'] == 'mean':
        for col in numerical_columns:
            data[col].fillna(data[col].mean(), inplace=True)
    elif options['null_handling'] == 'median':
        for col in numerical_columns:
            data[col].fillna(data[col].median(), inplace=True)
    elif options['null_handling'] == 'constant':
        data[numerical_columns].fillna(options['null_constant'], inplace=True)
    
    # Handle null values for categorical columns
    categorical_columns = data.select_dtypes(include='object').columns.tolist()
    categorical_columns = [col for col in categorical_columns if col not in exclude_columns]  # Exclude specified columns
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    if options['null_handling_categorical'] == 'drop':
        data.dropna(subset=categorical_columns, inplace=True)
    elif options['null_handling_categorical'] == 'mode':
        for col in categorical_columns:
            data[col].fillna(data[col].mode()[0], inplace=True)
    elif options['null_handling_categorical'] == 'constant':
        data[categorical_columns].fillna(options['null_categorical_constant'], inplace=True)
    
    # Scaling
    if options['scaling'] == 'standard':
        scaler = StandardScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        test[numerical_columns] = scaler.fit_transform(test[numerical_columns])
    elif options['scaling'] == 'minmax':
        scaler = MinMaxScaler()
        data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
        test[numerical_columns] = scaler.fit_transform(test[numerical_columns])

    if options['categorical_handling'] == 'onehot':
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
        test = pd.get_dummies(test, columns=categorical_columns, drop_first=True)
        # Align the test dataset columns with the training dataset
        # Drop the target column from data.columns
        data_columns_without_target = data.columns.drop(target_column)
        # Reindex test with the modified columns
        test = test.reindex(columns=data_columns_without_target, fill_value=0)

    elif options['categorical_handling'] == 'label':
        print('label encoding')
        label_encoders = {}
        for column in categorical_columns:
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
            test[column] = le.transform(test[column])
            label_encoders[column] = le  # Store label encoder for each column if needed later

    if data[target_column].dtype == 'object':
        le = LabelEncoder()
        data[target_column] = le.fit_transform(data[target_column])
        joblib.dump(le, f'{target_column}_label_encoder.pkl')
        print(f"Encoded target column '{target_column}' with LabelEncoder.")
    
    data.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    print("Preprocessing complete")


from flask import jsonify, request

@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.get_json()
    print(data)
    # Extract options from the JSON data
    options = {
        "null_handling": data.get('null_handling'),
        "null_constant": data.get('null_constant', None),  # Optional, for 'constant' option
        "null_handling_categorical": data.get('null_handling_categorical'),
        "null_categorical_constant": data.get('null_categorical_constant', 'Unknown'),  # Default to 'Unknown'
        "scaling": data.get('scaling'),
        "categorical_handling": data.get('categorical_handling'),
    }
    
    print(data.get('target'))   
    apply_preprocessing(options, data.get('target'))
    
    return jsonify({"message": "Data processed successfully"})

def train_model(model_type, target):
    data = pd.read_csv('train.csv')
    X = data.drop(target, axis=1)
    y = data[target]
    
    print('training model')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model based on the selected type
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor()
    elif model_type == 'knn':
        model = KNeighborsClassifier()
    elif model_type == 'logistic':
        model = LogisticRegression()
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    print('model trained')
    
    # Save the trained model to the local directory
    model_filename = f"model.pkl"
    joblib.dump(model, model_filename)
    return model_filename

@app.route('/model_selection', methods=['POST'])
def model_selection():
    print('model selection')
    data = request.get_json()
    model_type = data.get('model')
    print(model_type)
    model_file = train_model(model_type, data.get('target'))
    return jsonify({"message": "Model trained successfully", "model_file": model_file})

@app.route('/model_trained', methods=['GET'])
def model_trained():
    model = joblib.load('model.pkl')
    
    test_data = pd.read_csv('test.csv')
    predictions = model.predict(test_data)
    
    # Check if 'Id', 'ID', or 'Id' column exists and keep it along with predictions
    id_column = None
    for possible_id_column in ['Id', 'ID', 'id']:
        if possible_id_column in test_data.columns:
            id_column = possible_id_column
            break
    
    if id_column:
        final_data = test_data[[id_column]].copy()
    else:
        final_data = pd.DataFrame(index=test_data.index)
    
    final_data['predictions'] = predictions
    
    # Save the modified test set with only Id and predictions
    final_data.to_csv('final.csv')
    
    return jsonify({"message": "Model predictions generated successfully", "predictions_file": "final.csv"})


if __name__ == '__main__':
    app.run(debug=True)