import pickle
from flask import Flask, request, jsonify
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # data = request.get_json()

    # Extract the input values
    sodium = request.form.get("N")
    phosphorus = request.form.get("P")
    potassium = request.form.get("K")
    temperature = request.form.get("temperature")
    humidity = request.form.get("humidity")
    ph = request.form.get("ph")
    rainfall = request.form.get("rainfall")

    # Perform any necessary preprocessing on the input data
    # ...

    # Make the prediction
    input_data = np.array([[sodium, phosphorus, potassium, temperature, humidity, ph, rainfall]])

    #input_data = np.array([[40, 60, 30, 23, 60.3, 6.7, 140]])
    result = model.predict(input_data)[0]

    # return result

    return jsonify({'crop': str(result)})


if __name__ == '__main__':
    app.run(debug=True)  # See PyCharm help at https://www.jetbrains.com/help/pycharm/
