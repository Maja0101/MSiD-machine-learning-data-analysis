import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_set = pd.read_csv('AviationData.csv', encoding='ISO-8859-1', low_memory=False)
data_set['Event.Date'] = pd.to_datetime(data_set['Event.Date'], format="%Y-%m-%d")

min_year = 1982
X = np.array([x for x in range(min_year, 2023)])

y = [0 for i in range(min_year, 2023)]

for date in data_set['Event.Date']:
    if date.year >= min_year:
        y[date.year-min_year] += 1

y = np.array(y)

acccidents = pd.DataFrame(list(zip(X, y)), columns=['Event_Date', 'Accidents_Number'])

matrix = np.array(acccidents.values,'int')

X = matrix[:,0]
y = matrix[:,1]

X = X/(np.max(X)) 

m = np.size(y)
X = X.reshape([m,1])
x = np.hstack([np.ones_like(X),X])

def computecost(theta):
    return (1/(2*m)) * (np.sum(((x@theta)-y)**2))

def gradient_descent(theta, alpha = 0.0001, iteration = 2000):
    m = np.size(y)
    J_history = np.zeros([iteration, 1])
    for i in range(iteration):
        errors = (x @ theta) - y
        temp0 = theta[0] - ((alpha/m) * np.sum(errors*x[:,0]))
        temp1 = theta[1] - ((alpha/m) * np.sum(errors*x[:,1]))
        theta = np.array([temp0,temp1]).reshape(2,1)
        J_history[i] = computecost(theta)

    return theta, J_history

theta = np.zeros([2,1])
print(theta,'\n',m)
print(computecost(theta))
print("-----")

theta, J = gradient_descent(theta)
print(theta)
print(J)

plt.plot(X,y,'bo')
plt.plot(X,x@theta,'-')
plt.axis([0,1,3,7])
plt.ylabel('Accidents_Number')
plt.xlabel('Event_Date')
plt.legend(['Accidents','LinearFit'])
plt.title('Accidents in years')
plt.grid()
plt.show()