import pickle

from flask import Flask
from flask import render_template
from flask import request
from flask import flash

model_name = 'model.pkl'

with open("../models/{}".format(model_name), 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts the class of the inputted image
    """
    if request.method == 'POST':
        # Get the data from the form
        data = request.form
        try:
            sepal_length = float(data['sepal_length'])
            sepal_width = float(data['sepal_width'])
            petal_length = float(data['petal_length'])
            petal_width = float(data['petal_width'])
        except Exception:
            return render_template('index.html')

        # Make a prediction
        prediction = model.predict(
            [[sepal_length, sepal_width, petal_length, petal_width]])[0]

        result = ""
        if prediction == 0:
            result = "Iris-setosa"
        elif prediction == 1:
            result = "Iris-versicolor"
        else:
            result = "Iris-virginica"

        # Flash the message to the user
        flash("The predicted class is {}".format(result))

        # Return the prediction
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=3001)
