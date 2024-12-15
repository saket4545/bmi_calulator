from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Define the KerasCustomClassifier class
class KerasCustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=100, batch_size=32):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.create_model()

    def create_model(self):
        model = Sequential([
            Dense(36, activation='relu', input_shape=(3,)),
            Dense(24, activation='relu'),
            Dense(12, activation='relu'),
            Dense(6, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return self

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=-1)

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]

# Load the pre-trained pipeline
with open('bmi_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# BMI Categories
categories = {
    0: "Extremely Weak",
    1: "Weak",
    2: "Normal",
    3: "Overweight",
    4: "Obesity",
    5: "Extreme Obesity"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    gender = request.form['gender']
    height = float(request.form['height'])
    weight = float(request.form['weight'])

    # Prepare the input data for prediction
    input_data = pd.DataFrame([[gender, height, weight]], columns=['Gender', 'Height', 'Weight'])

    # Use the model to predict the BMI index
    prediction = pipeline.predict(input_data)

    # Convert prediction to category label
    category = categories[prediction[0]]

    return render_template('index.html', prediction_text=f'Your BMI category is: {category}')

if __name__ == '__main__':
    app.run(debug=True)
