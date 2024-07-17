from flask import Blueprint, render_template, request, jsonify
import joblib
import os
from .model.network import NeuralNetwork

main = Blueprint('main', __name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'model_weights.pkl')
model = NeuralNetwork()
model.load_weights(model_path)

@main.route('/')
def index():
    return "Index"

@main.route('/predict', methods=['POST'])
def predict():
    return "Prediction route"
