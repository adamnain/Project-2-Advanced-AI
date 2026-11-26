import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.classes = None
        self.cost_history = []

    def softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def categorical_cross_entropy(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-12, 1.0)
        return float(-np.mean(np.sum(y * np.log(y_hat), axis=1)))

    def fit(self, X, y):
        m, n = X.shape

        # Inisialisasi
        self.classes, inverse = np.unique(y, return_inverse=True)
        self.weights = np.zeros((n, len(self.classes)))
        self.bias = np.zeros(len(self.classes))

        # One Hot Encoding
        y_one_hot = np.zeros((y.size, self.classes.size))
        y_one_hot[np.arange(y.size), inverse] = 1

        for _ in range(self.epochs):
            # Forward Propagation
            z = np.dot(X, self.weights) + self.bias
            y_hat = self.softmax(z)

            # Loss
            self.cost_history.append(self.categorical_cross_entropy(y_hat, y_one_hot))

            # Gradient Descent
            dW = (np.dot(X.T, (y_hat - y_one_hot))) / m
            db = (np.sum(y_hat - y_one_hot, axis=0)) / m

            # Update Weight dan Bias
            self.weights -= self.lr * dW
            self.bias -= self.lr * db

        return self

    # Prediksi probabilitas
    def predict_probabilities(self, X):
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        z = np.dot(X, self.weights) + self.bias
        return self.softmax(z)

    # Prediksi class
    def predict(self, X):
        probabilities = self.predict_probabilities(X)
        return self.classes[np.argmax(probabilities, axis=1)]