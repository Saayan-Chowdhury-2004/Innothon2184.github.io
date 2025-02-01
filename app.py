import numpy as np
import pandas as pd
from tensorflow.keras.models import model_from_json
from flask import Flask, request, render_template
import pickle

app = Flask(_name_)

def preprocess_data(data):
    df_processed = data.copy()
    df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    return df_processed

def create_sequences(data, steps=33):
    X, y = [], []
    if len(data) < steps:
        raise ValueError("Not enough data to create sequences. Need at least {} rows.".format(steps))
    for i in range(len(data) - steps):
        X.append(data[i:i + steps])
        y.append(data[i + steps])
    return np.array(X), np.array(y)

def feature_fix_shape(train, test):
    train = train.reshape(train.shape[0], train.shape[1], 1)
    test = test.reshape(test.shape[0], test.shape[1], 1)
    return train, test

def load_model_and_predict(model_file, hyperparams_file, weights_file, new_data, historical_data, num_days):
    with open(model_file, "r") as f:
        model_json = f.read()
    model = model_from_json(model_json)
    
    with open(hyperparams_file, "rb") as f:
        best_hyperparameters = pickle.load(f)
    
    model.load_weights(weights_file)
    
    combined_data = pd.concat([historical_data, new_data], ignore_index=True)
    df_processed = preprocess_data(combined_data)
    
    try:
        X_new, _ = create_sequences(df_processed.values)
    except ValueError as e:
        print("Error in sequence creation:", e)
        return []
    
    try:
        X_new, _ = feature_fix_shape(X_new, X_new)
    except IndexError as e:
        print("Error reshaping X_new:", e)
        return []
    
    predictions = []
    current_input = X_new[-1:]
    
    for _ in range(num_days):
        next_prediction = model.predict(current_input)
        predictions.append(next_prediction[0, 0])
        current_input = np.roll(current_input, -1, axis=1)
        current_input[:, -1, 0] = next_prediction[0, 0]
    
    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []
    if request.method == "POST":
        new_data_value = float(request.form['new_data'])
        historical_data = pd.DataFrame({
            "DateTime": pd.date_range(start="2024-06-01", periods=40, freq="D"),
            "Junction": [440] * 40,
            "Vehicles": np.random.randint(50, 100, size=40),
            "ID": [12345] * 40
        })
        new_data = pd.DataFrame({
            "DateTime": [pd.Timestamp("2024-06-30 12:00:00")],
            "Junction": [141],
            "Vehicles": [new_data_value],
            "ID": [12345]
        })
        
        model_file = "model.json"
        hyperparams_file = "best_hyperparameters.pkl"
        weights_file = "model_weights.h5"
        
        predictions = load_model_and_predict(model_file, hyperparams_file, weights_file, new_data, historical_data, num_days=10)
    
    return render_template('index.html', predictions=predictions)

if _name_ == "_main_":
    app.run(debug=True)
