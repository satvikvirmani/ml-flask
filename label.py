import joblib
from network import NeuralNetwork
import os

os.chdir(os.getcwd() + "/model")

def label() -> int:
    load_net = joblib.load('model.pkl')
    return load_net.test_prediction(0)