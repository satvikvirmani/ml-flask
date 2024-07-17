import numpy as np
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("test.csv")
data = np.array(data)
m_dat, n_dat = data.shape
data_train = data[0 : m_dat].T
X_train = data_train[0 : m_dat]
n_train, m_train = X_train.shape
X_train = X_train / 255.

for i in range(1, 5):
    current_image = X_train[:, i].reshape(28, 28)
    current_image = (current_image * 255).astype(np.uint8)
    new_image = Image.fromarray(current_image, mode='L')
    new_image.save(f'Image{i}.png')
