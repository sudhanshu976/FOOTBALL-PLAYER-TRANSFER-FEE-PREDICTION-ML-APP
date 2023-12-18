from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import joblib
from scipy.stats import boxcox

app = Flask(__name__)

# Load the pre-trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)


scaler= joblib.load("scaler.joblib")
lambda_value = 0.4469754727550972
teams = ['Al-Batin FC', 'Botafogo de Futebol e Regatas', 'Daegu FC', 'Daejeon Hana Citizen', 
         'FC Seoul', 'FC Tokyo', 'Gangwon FC', 'Incheon United', 'Jeonbuk Hyundai Motors',
         'Kashiwa Reysol', 'Mamelodi Sundowns FC', 'Marumo Gallants FC', 'Other',
         'Royal AM FC', 'Sagan Tosu', 'Santos FC', 'Suwon Samsung Bluewings',
         'Swallows FC', 'São Paulo Futebol Clube', 'Vissel Kobe', 'Yokohama FC']

# Pre-process the teams list to include index and label
teams_with_index = [{'index': idx, 'label': label} for idx, label in enumerate(teams)]
# # Dummy data for one-hot encoding
# teams = ['Al-Batin FC', 'Botafogo de Futebol e Regatas', 'Daegu FC', 'Daejeon Hana Citizen', 
#          'FC Seoul', 'FC Tokyo', 'Gangwon FC', 'Incheon United', 'Jeonbuk Hyundai Motors',
#          'Kashiwa Reysol', 'Mamelodi Sundowns FC', 'Marumo Gallants FC', 'Other',
#          'Royal AM FC', 'Sagan Tosu', 'Santos FC', 'Suwon Samsung Bluewings',
#          'Swallows FC', 'São Paulo Futebol Clube', 'Vissel Kobe', 'Yokohama FC']

@app.route('/')
def home():
    return render_template('index.html', teams=teams_with_index)

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    age = int(request.form['age'])
    current_value = float(request.form['current_value'])
    team = int(request.form['team'])
    position = int(request.form['position'])
    appearance_median = float(request.form['appearance_median'])
    minutes_played_boxcox = float(request.form['minutes_played_boxcox'])
    games_injured_binary = int(request.form['games_injured_binary'])
    award_binary = int(request.form['award_binary'])

    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'age': [age],
        'current_value': [current_value],
        'team': [team],
        'position': [position],
        'appearance_median': [appearance_median],
        'minutes_played_boxcox': [minutes_played_boxcox],
        'games_injured_binary': [games_injured_binary],
        'award_binary': [award_binary]
    })
    print(user_data)
    print(type(user_data))
    user_data['minutes_played_boxcox'] = boxcox(user_data['minutes_played_boxcox'] + 1, lmbda=lambda_value)
    print(user_data)
    # 
    user_array = np.array(user_data)
    print(user_array)

    user_array_standardized = scaler.transform(user_array)
    print(user_array_standardized)
    # Make a prediction
    prediction = model.predict(user_array_standardized)
    prediction= int(prediction[0])
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0' , port=8080)
