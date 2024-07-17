from flask import Flask, Blueprint, jsonify, request
from joblib import load
from network import NeuralNetwork
import numpy as np
from PIL import Image
    
app = Flask(__name__)

net = NeuralNetwork()
model = load("dumped.joblib")

@app.route('/')
def index():
    return "Index"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    img = Image.open(file.stream)
    
    if img.size != (28, 28):
        return jsonify({"error": "Image size must be 28x28 pixels"}), 400
    
    img = img.convert('L')
    img = np.asarray(img)
    img = img.flatten().reshape(784, 1)
    label = model.test_prediction(img)
    return jsonify({'message': 'success', 'label': str(label)})

if __name__ == '__main__':
    app.run(debug=True)