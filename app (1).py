import pickle
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Assuming the input data is a dictionary with keys matching your model's input features
        # You may need to adjust this based on your actual input data format

        # Preprocess the input data (if necessary) - e.g., label encoding, feature scaling
        # ...

        # Make a prediction using your model
        input_features = pd.DataFrame([data])  # Convert input to a DataFrame
        prediction = model.predict(input_features)

        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
