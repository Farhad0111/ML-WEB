#app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Sample data for pizza details
pizza_details = {
    "chicken": {
        "image": "static/Pizza/chicken.jpeg",
        "price": "$12.99",
        "ingredients": ["Chicken", "Cheese", "Tomato Sauce", "Bell Peppers"],
        "reviews": 4.5,
        "sales": 123
    },
    "classic": {
        "image": "static/Pizza/classic.jpg",
        "price": "$10.99",
        "ingredients": ["Cheese", "Tomato Sauce", "Basil"],
        "reviews": 4.2,
        "sales": 89
    },
    "supreme": {
        "image": "static/Pizza/supreme.jpg",
        "price": "$15.99",
        "ingredients": ["Pepperoni", "Sausage", "Cheese", "Bell Peppers", "Olives"],
        "reviews": 4.7,
        "sales": 150
    },
    "veggie": {
        "image": "static/Pizza/veggie.jpg",
        "price": "$11.99",
        "ingredients": ["Mushrooms", "Bell Peppers", "Onions", "Tomato Sauce"],
        "reviews": 4.3,
        "sales": 75
    }
}

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
    pizza_types = {0: 'chicken', 1: 'classic', 2: 'supreme', 3: 'veggie'}
    predicted_type = pizza_types.get(prediction, "Unknown")

    # Render result page with the prediction and confidence level
    #return render_template('prediction.html', prediction=predicted_type, confidence=confidence)
    return render_template('result.html', prediction=predicted_type, confidence=confidence)

@app.route('/details/<pizza_type>')
def pizza_details_page(pizza_type):
    pizza = pizza_details.get(pizza_type)
    if pizza:
        return render_template('details.html', pizza=pizza, pizza_type=pizza_type)
    else:
        return "Pizza not found", 404

@app.route('/order/<pizza_type>')
def order_page(pizza_type):
    pizza = pizza_details.get(pizza_type)
    if pizza:
        return render_template('order.html', pizza=pizza, pizza_type=pizza_type)
    else:
        return "Pizza not found", 404

if __name__ == '__main__':
    app.run(debug=True)


