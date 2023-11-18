# iris_prediction_flask.py

from flask import Flask, request, jsonify
import joblib
from sklearn.datasets import load_iris

# Load the Iris dataset to get class names
iris = load_iris()
class_names = iris.target_names

# Load the trained model
model_filename = 'iris_model.joblib'
clf = joblib.load(model_filename)

# Create a Flask web application
app = Flask(__name__)

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the HTTP request
        input_data = request.get_json()

        # Make predictions
        predictions = clf.predict([input_data['data']])

        # Map class numbers to class names
        predicted_class_names = [class_names[i] for i in predictions]

        # Prepare the response
        response = {
            'prediction': predicted_class_names[0],
            'confidence': max(clf.predict_proba([input_data['data']])[0])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the application on port 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
