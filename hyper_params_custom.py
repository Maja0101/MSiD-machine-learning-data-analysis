import numpy as np
from itertools import product
from sklearn.metrics import accuracy_score
from utils import prepare_data
from CustomLogisticRegressionL1 import CustomLogisticRegressionL1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

X, y = prepare_data(more_data=True)

tl = TomekLinks(sampling_strategy='majority')
X, y = tl.fit_resample(X, y)

degree = 2

train_scores = {}
test_scores = {}
avg_train_scores = {}
avg_test_scores = {}
times = {}

class_rep = None
conf_mx = None

learning_rates = [0.0001, 0.001, 0.01, 0.1]
num_iterations = [100, 500, 1000]
reg_strengths = [0.0, 0.0001, 0.001, 0.01]

best_params = None
best_score = -np.inf
best_avg_params = None
best_avg_score = -np.inf

poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

scaler = StandardScaler()
X_poly = scaler.fit_transform(X_poly)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=39)

print("preparation finished")

for lr, iters, reg in product(learning_rates, num_iterations, reg_strengths):
    print(lr, iters, reg)
    train_scores[(lr, iters, reg)] = []
    test_scores[(lr, iters, reg)] = []
    times[(lr, iters, reg)] = []
    for train_index, test_index in skf.split(X_poly, y):
        print("dividing data")
        X_train_fold, X_test_fold = X_poly[train_index], X_poly[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        print("creating and trainig model")

        model = CustomLogisticRegressionL1(lr=lr, epochs=iters, lambda_L1=reg)

        start_time = time()
        model.fit(X_train_fold, y_train_fold, X_test_fold, y_test_fold)
        end_time = time()

        print("end of training")

        y_pred = model.predict(X_test_fold)
        score = accuracy_score(y_test_fold, y_pred)
        train_score = model.score(X_train_fold, y_train_fold)
        train_scores[(lr, iters, reg)].append(train_score)
        test_scores[(lr, iters, reg)].append(score)
        times[(lr, iters, reg)].append(end_time - start_time)

        print("Train:", train_score)
        print("Test:", score)
        
        if score > best_score:
            best_score = score
            best_params = (lr, iters, reg)
            class_rep = classification_report(y_test_fold, y_pred, zero_division=np.nan)
            conf_mx = confusion_matrix(y_test_fold, y_pred)
    
    avg_train_scores[(lr, iters, reg)] = np.mean(train_scores[(lr, iters, reg)])
    avg_test_scores[(lr, iters, reg)] = np.mean(test_scores[(lr, iters, reg)])

    if avg_test_scores[(lr, iters, reg)] > best_avg_score:
        best_avg_score = avg_test_scores[(lr, iters, reg)]
        best_avg_params = (lr, iters, reg)

print("Best parameters:", best_params)
print("Best score:", best_score)
print("······ Clasification Report ······")
print(class_rep)
print("······ Confusion Matrix  ······")
print(conf_mx)

print("Details")
for key, val in times.items():
    print(key)
    print("Time:", val)
    print("Train scores:", train_scores[key])
    print("Test scores:", test_scores[key])
    print("Avg train score:", avg_train_scores[key])
    print("Avg test score:", avg_test_scores[key])
