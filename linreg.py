import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, accuracy_score

def prepare_data():
    data_set = pd.read_csv('AviationData.csv', encoding='ISO-8859-1', low_memory=False)
    data_set['Event.Date'] = pd.to_datetime(data_set['Event.Date'], format="%Y-%m-%d")

    min_year = 1982
    X = np.array([x for x in range(min_year, 2023)])

    y = [0 for i in range(min_year, 2023)]

    for date in data_set['Event.Date']:
        if date.year >= min_year:
            y[date.year-min_year] += 1

    y = np.array(y)

    return X, y

def model(parameters, x):
    a, b = parameters
    return a * x + b

def analytic(X, y):
    x_mean = np.average(X)
    y_mean = np.average(y)
    a_est = np.sum((X-x_mean) * (y-y_mean)) / np.sum((X-x_mean)**2)
    b_est = y_mean - a_est * x_mean

    X_test = np.linspace(start=X.min(), stop=X.max(), num=2025)
    y_pred = model(parameters=[a_est, b_est], x=X_test)

    plt.scatter(X,y)
    plt.plot(X_test, y_pred, color='orange')
    plt.xlabel('x - year', fontsize=14)
    plt.ylabel('y - number of accidents', fontsize=14)
    plt.title(f"y = {a_est} * x + {b_est}")
    plt.show()

    return a_est, b_est

def closed_form_solution(X, y):
    _X = X[np.newaxis, :] ** [[1], [0]]
    _Y = y[np.newaxis, :]
    # print(f"_X: {_X.shape}, _Y: {_Y.shape}")

    _T = np.linalg.inv(_X @ _X.transpose()) @ _X @ _Y.transpose()
    # print(f"_T: {_T.ravel()}")

    def plot_fig(X: np.ndarray, Y: np.ndarray, coeff: np.ndarray):
        X_test = np.linspace(start=X.min(), stop=X.max(), num=2025)
        func_str = "y = "
        Y_pred = model(coeff, X_test)
        for i, c in enumerate(coeff.ravel()[::-1]):
            func_str += f"{round(c, 4)} * x ** {i} + "

        plt.scatter(X, Y, label='real data')
        plt.plot(X_test, Y_pred, color='orange', label='estimated trend')
        plt.xlabel('x - year', fontsize=14)
        plt.ylabel('y - number of accidents', fontsize=14)
        plt.title(f"Matched function: {func_str[:-2]}")
        plt.legend()
        plt.show()

    plot_fig(X, y, _T)

    return _T.ravel()[0], _T.ravel()[1]

def draw_errors(X, y, a_est, b_est):
    Y_ = model(parameters=[a_est, b_est], x=X)
    E  = abs(y - Y_)
    err = np.sqrt(np.sum(E ** 2)) / y.size

    X_test = np.linspace(start=X.min(), stop=X.max(), num=2025)
    Y_pred = model(parameters=[a_est, b_est], x=X_test)
    
    plt.errorbar(X, Y_, yerr=E, color='orange', capsize = 5)
    plt.fill_between(X_test, Y_pred - err, Y_pred + err, color='orange', alpha=0.2)

    plt.scatter(X,y)
    plt.plot(X_test, Y_pred, color='orange')
    plt.xlabel('x - year', fontsize=14)
    plt.ylabel('y - number of accidents', fontsize=14)
    plt.show()

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def compare(X, y):
    a, b = closed_form_solution(X, y)
    y_eval_closed_form = a * X.ravel() + b
    rmse_closed_form = rmse(y, y_eval_closed_form)

    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    regr = LinearRegression() 
    regr.fit(X, y) 
    y_eval_sklearn = regr.predict(X)

    rmse_sklearn = rmse(y, y_eval_sklearn)

    print("\n\n······ Linear regression using Scikit-learn ······")
    print(f"y = {regr.coef_[0][0]} * x + {regr.intercept_[0]}")
    print(f"RMSE = {rmse_sklearn}")

    print("\n\n······ Linear regression using closed form solution ······")
    print(f"y = {a} * x + {b}")  
    print(f"RMSE = {rmse_closed_form}") 



if __name__ == "__main__":
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)
    compare(X, y)
    compare(X_train, y_train)
    compare(X_test, y_test)

