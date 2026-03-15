import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import prepare_data

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / np.sum(e_z, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    y_pred = softmax(y_pred)
    loss = 0

    for i in range(len(y_pred)):
        loss = loss + (-1 * y_true[i]*np.log(y_pred[i]))

    return loss

def gradient(X, y_true, y_pred, n_classes):
    m = X.shape[0]

    y_one_hot = np.zeros((m, n_classes))
    y_one_hot[np.arange(m), y_true.astype(int)] = 1

    dz = y_pred - y_one_hot
    dW = X.T @ dz / m
    db = np.sum(dz, axis=0) / m

    return dW, db

def graddesc_logreg(X, y, lr=0.1, epochs=1000, batch_size=32):
    m, n = X.shape
    num_of_classes = int(np.max(y)) + 1

    W = np.zeros((n, num_of_classes))
    b = np.zeros((num_of_classes,))

    for epoch in range(epochs):
        indexes = np.arange(m)
        np.random.shuffle(indexes)
        X = X[indexes]
        y = y[indexes]

        for i in range(0, m, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            y_pred = softmax(X_batch @ W + b)

            loss = cross_entropy(y_pred, y_batch)

            dW, db = gradient(X_batch, y_batch, y_pred, num_of_classes)

            W -= lr * dW
            b -= lr * db
        
        # if epoch % 10 == 0 or epoch == epochs - 1:
        #     print(f"Epoch {epoch}: loss = {loss}")
    
    return W, b

def compare_methods():
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

    # W, b = graddesc_logreg(X_train, y_train)
    # print(W, b)

    # results from graddesc_log
    W = np.array([[-0.6436977,-2.06578182,-0.39203057,3.10151009]])
    b = np.array([-0.89071832,-1.02424617,2.08435948,-0.16939499])
    # print(W, b)

    y_eval_train = softmax(X_train @ W + b)
    y_eval_train_graddesc = np.argmax(y_eval_train, axis=1)

    y_eval_graddesc = softmax(X_test @ W + b)
    y_eval_graddesc = np.argmax(y_eval_graddesc, axis=1)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)
    y_eval_sklearn = model.predict(X_test)
    y_eval_train_sklearn = model.predict(X_train)

    print("Gradient descent accurancy:", accuracy_score(y_test, y_eval_graddesc))
    print("Gradient descent train accurancy:", accuracy_score(y_train, y_eval_train_graddesc))
    print("Scikit-learn accurancy:", accuracy_score(y_test, y_eval_sklearn))
    print("Scikit-learn train accurancy:", accuracy_score(y_train, y_eval_train_sklearn))

    print("\n\n······ Clasification Report for Gradient Descent ······")
    print(classification_report(y_test, y_eval_graddesc, zero_division=np.nan))

    print("\n\n······ Clasification Report for train Gradient Descent ······")
    print(classification_report(y_train, y_eval_train_graddesc, zero_division=np.nan))

    print("\n\n······ Clasification Report for Scikit-learn ······")
    print(classification_report(y_test, y_eval_graddesc, zero_division=np.nan))

    print("\n\n······ Clasification Report for train Scikit-learn ······")
    print(classification_report(y_train, y_eval_train_sklearn, zero_division=np.nan))

    print("\n\n······ Confusion Matrix for Gradient Descent ······")
    print(confusion_matrix(y_test, y_eval_graddesc))

    print("\n\n······ Confusion Matrix for train Gradient Descent ······")
    print(confusion_matrix(y_train, y_eval_train_graddesc))

    print("\n\n······ Confusion Matrix for Scikit-learn ······")
    print(confusion_matrix(y_test, y_eval_sklearn))

    print("\n\n······ Confusion Matrix for train Scikit-learn ······")
    print(confusion_matrix(y_train, y_eval_train_sklearn))

if __name__ == "__main__":
    compare_methods()
