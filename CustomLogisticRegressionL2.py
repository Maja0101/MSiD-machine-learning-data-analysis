import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class CustomLogisticRegressionL2:
    def __init__(self, lr=0.1, epochs=1000, batch_size=32, lambda_L2=0.0):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.epsilon = 1e-15
        self.lambda_L2  = lambda_L2

    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)
    
    def cross_entropy(self, l, y_true):
        pred = self.softmax(l)
        pred = np.clip(pred, self.epsilon, 1 - self.epsilon)

        loss = -np.sum(y_true * np.log(pred)) / y_true.shape[0]

        l2_penalty = (self.lambda_L2 / 2) * np.sum(self.W ** 2)

        return loss + l2_penalty
    
    def gradient(self, X, y_true, l, n_classes):
        m = X.shape[0]
        pred = self.softmax(l)

        y_one_hot = self._one_hot(y_true, n_classes)

        dz = pred - y_one_hot
        dW = (X.T @ dz) / m
        dW += self.lambda_L2 * self.W
        db = np.sum(dz, axis=0) / m

        return dW, db
    
    def fit(self, X, y, X_test=None, y_test=None):
        m, n = X.shape
        self.num_of_classes = int(np.max(y)) + 1

        self.W = np.zeros((n, self.num_of_classes))
        self.b = np.zeros((self.num_of_classes,))

        self.train_losses = []
        self.test_losses = []

        for epoch in range(self.epochs):
            indexes = np.random.permutation(m)

            for i in range(0, m, self.batch_size):
                batch_indexes = indexes[i:i + self.batch_size]
                X_batch = X[batch_indexes]
                y_batch = y[batch_indexes]
                m_batch = X_batch.shape[0]

                logits = X_batch @ self.W + self.b

                dW, db = self.gradient(X_batch, y_batch, logits, self.num_of_classes)

                self.W -= self.lr * dW
                self.b -= self.lr * db

            logits_train = X @ self.W + self.b
            train_loss = self.cross_entropy(logits_train, self._one_hot(y, self.num_of_classes))
            self.train_losses.append(train_loss)

            if X_test is not None and y_test is not None:
                logits_test = X_test @ self.W + self.b
                test_loss = self.cross_entropy(logits_test, self._one_hot(y_test, self.num_of_classes))
                self.test_losses.append(test_loss)

        return self

    def predict(self, X):
        y_pred = self.softmax(X @ self.W + self.b)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        return self.softmax(X @ self.W + self.b)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def _one_hot(self, y, n_classes):
        m = y.shape[0]
        y_one_hot = np.zeros((m, n_classes))
        y_one_hot[np.arange(m), y.astype(int)] = 1
        return y_one_hot