# iris_prediction.py

# Import necessary libraries
import joblib
import sys
from sklearn.datasets import load_iris

# Load the Iris dataset to get class names
iris = load_iris()
class_names = iris.target_names

# Load the trained model
model_filename = 'iris_model.joblib'
clf = joblib.load(model_filename)

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 5:
    print("Usage: python iris_prediction.py sepal_length sepal_width petal_length petal_width")
    sys.exit(1)

# Get input from the terminal arguments
sepal_length, sepal_width, petal_length, petal_width = map(float, sys.argv[1:5])

# Example input for prediction
sample_input = [[sepal_length, sepal_width, petal_length, petal_width]]

# Make predictions
predictions = clf.predict(sample_input)

# Map class numbers to class names
predicted_class_names = [class_names[i] for i in predictions]

# Display the predictions
print(f"{predicted_class_names}")
