import numpy as np
import joblib

class NeuralNetwork:
    def __init__(self) -> None:
        self.n_layers = 0
        self.layers = dict()
        self.A = dict()
        self.Z = dict()
        self.derivatives = dict()
        pass
    
    def init_data(self, X, Y):
        self.X = X
        self.Y = Y
        self.n, self.m = X.shape

    def init_params(self, n_unit, n_weight):
        W = np.random.rand(n_unit, n_weight) - 0.5
        b = np.random.rand(n_unit, 1) - 0.5
        return W, b

    def layer(self, n_unit, n_weight):
        W, b = self.init_params(n_unit, n_weight)
        self.layers[self.n_layers] = {"W": W, "b": b}
        self.n_layers += 1

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def ReLU_derive(self, Z):
        return Z > 0

    def softmax(self, Z):
        Z_shifted = Z - np.max(Z, axis=0)
        exp_Z = np.exp(Z_shifted)
        return exp_Z / np.sum(exp_Z, axis=0)

    def forward_pass(self, X):
        dat_in = X
        for l in range(self.n_layers - 1):
            dat_in = self.layers[l]["W"].dot(dat_in) + self.layers[l]["b"]
            self.Z[l] = dat_in
            dat_in = self.ReLU(dat_in)
            self.A[l] = dat_in

        l += 1
        dat_in = self.layers[l]["W"].dot(dat_in) + self.layers[l]["b"]
        self.Z[l] = dat_in
        dat_in = self.softmax(dat_in)
        self.A[l] = dat_in

        return dat_in

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def clip_gradients(self, threshold=1.0):
        for l in range(self.n_layers):
            np.clip(self.derivatives[l]['dW'], -threshold, threshold, out=self.derivatives[l]['dW'])
            np.clip(self.derivatives[l]['db'], -threshold, threshold, out=self.derivatives[l]['db'])

    def back_prop(self, alpha):
        m_size = 4
        one_hot_y = self.one_hot(self.Y)

        l = self.n_layers - 1

        dZ = self.A[l] - one_hot_y
        dW = 1 / self.m * dZ.dot(self.A[l - 1].T)
        db = 1 / self.m * np.sum(dZ, axis=1)

        self.derivatives[l] = {"dZ": dZ, "dW": dW, "db": db}
        self.layers[l]["W"] += -alpha * dW
        self.layers[l]["b"] += -alpha * db.reshape(db.shape[0], 1)

        for l in range(self.n_layers - 2, 0 , -1):
            dZ = self.layers[l + 1]["W"].T.dot(self.derivatives[l + 1]["dZ"]) * self.ReLU_derive(self.Z[l])
            dW = 1 / self.m * dZ.dot(self.A[l - 1].T)
            db = 1 / self.m * np.sum(dZ, axis=1)
            self.derivatives[l] = {"dZ": dZ, "dW": dW, "db": db}
            self.layers[l]["W"] += -alpha * dW
            self.layers[l]["b"] += -alpha * db.reshape(db.shape[0], 1)

        l -= 1
        dZ = self.layers[l + 1]["W"].T.dot(self.derivatives[l + 1]["dZ"]) * self.ReLU_derive(self.Z[1 - 1])
        dW = 1 / self.m * dZ.dot(self.X.T)
        db = 1 / self.m * np.sum(dZ, axis=1)
        self.derivatives[l] = {"dZ": dZ, "dW": dW, "db": db}
        self.layers[l]["W"] += -alpha * dW
        self.layers[l]["b"] += -alpha * db.reshape(db.shape[0], 1)

        self.clip_gradients()

    def gradient_descent(self, alpha, iterations):
        for i in range(iterations):
            self.forward_pass(self.X)
            self.back_prop(alpha)
            if i % 10 == 0:
                print(f"iteration {i}")
                predictions = self.get_predictions()
                print(self.get_accuracy(predictions))

    def get_predictions(self):
        return np.argmax(self.A[self.n_layers - 1], 0)

    def get_accuracy(self, predictions):
        return np.sum(predictions == self.Y) / self.Y.size

    def make_predictions(self, X):
        self.forward_pass(X)
        predictions = self.get_predictions()
        return predictions

    def test_prediction(self, index) -> int:
        current_image = self.X[:, index, None]
        prediction = self.make_predictions(self.X[:, index, None])
        label = self.Y[index]
        return prediction[0]
