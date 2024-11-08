from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the pre-trained model
with open("DecisionTreeClassifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    try:
        input_data = [
            int(request.form['name']),  # Pizza name as an integer
            int(request.form['size']),  # Size as an integer
            int(request.form['price']), # Price as an integer
            int(request.form['month']), # Month as an integer
            int(request.form['hour'])   # Hour as an integer
        ]
    except ValueError:
        return "Please enter valid integer values."

    # Predict the pizza type
    prediction = model.predict([input_data])[0]

    # Get confidence level
    confidence = model.predict_proba([input_data]).max() * 100

    # Convert prediction to a human-readable format
    pizza_types = {0: 'Veg', 1: 'Non-Veg', 2: 'Vegan', 3: 'Special'}
    predicted_type = pizza_types.get(prediction, "Unknown")

    # Render result page with the prediction and confidence level
    return render_template('result.html', prediction=predicted_type, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
