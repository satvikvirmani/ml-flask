import numpy as np
import pandas as pd
import os
from joblib import dump
from network import NeuralNetwork

data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/train.csv"))
data = np.array(data)
m_dat, n_dat = data.shape # 42000, 785
np.random.shuffle(data)

validation_set = int(m_dat / 10)

data_validation = data[0 : validation_set].T
data_train = data[validation_set : m_dat].T
X_val = data_validation[1 : m_dat]# 784 x 4200
Y_val = data_validation[0] # 4200 x _
n_val, m_val = X_val.shape
X_val = X_val / 255.

X_train = data_train[1 : m_dat]# 784 x 4200
Y_train = data_train[0] # 4200 x _
n_train, m_train = X_train.shape
X_train = X_train / 255.

np.set_printoptions(precision=3)

net = NeuralNetwork()
net.init_data(X_train, Y_train)
net.layer(20, 784) # W: 4 x 3, b: 4 x 1
net.layer(15, 20) # W: 3 X 4, b: 3 x 1
net.layer(10, 15) # W: 2 X 3, b: 2 x 1
net.gradient_descent(0.01, 500)

model = dump(net, "dumped.joblib")
