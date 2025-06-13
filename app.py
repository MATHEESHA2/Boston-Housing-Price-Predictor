from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    with open('house_price_prediction.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print("Error loading model:", e)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
                     'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

    try:
        # Collect input features
        features = [float(request.form.get(name)) for name in feature_names]

        # Predict using model
        prediction = model.predict([features])[0]
        predicted_price = f"🏠 Estimated Price: ${prediction:,.2f}"

        return render_template('index.html', prediction=predicted_price)

    except ValueError:
        return render_template('index.html', prediction="⚠️ Please enter valid numeric values in all fields.")
    except Exception as e:
        return render_template('index.html', prediction=f"❌ An unexpected error occurred: {str(e)}")

# Start the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=81)
