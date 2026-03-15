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
from MixtureOfExperts import MixtureOfExpertsSoft, MixtureOfExpertsHard, MixtureOfExpertsSoftV2

X, y = prepare_data(more_data=True)

tl = TomekLinks(sampling_strategy='majority')
X, y = tl.fit_resample(X, y)

degree = 2

train_scores = []
test_scores = []
avg_train_score = 0.0
avg_test_score = 0.0
times = []

class_reps = []
conf_mxs = []

poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)

scaler = StandardScaler()
X_poly = scaler.fit_transform(X_poly)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=39)

print("preparation finished")

for train_index, test_index in skf.split(X_poly, y):
    print("dividing data")
    X_train_fold, X_test_fold = X_poly[train_index], X_poly[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    print("creating and trainig model")

    model = CustomLogisticRegressionL1(lr=0.001, epochs=100, lambda_L1=0.0001)
    # model = CustomLogisticRegressionL1(lr=0.01, epochs=500, lambda_L1=0.001)
    # model = CustomLogisticRegressionL1(lr=0.0001, epochs=1000, lambda_L1=0.0)

    start_time = time()
    model.fit(X_train_fold, y_train_fold)
    end_time = time()

    print("end of training")

    y_pred = model.predict(X_test_fold)

    test_scores.append(accuracy_score(y_test_fold, y_pred))
    train_scores.append(model.score(X_train_fold, y_train_fold))
    times.append(end_time - start_time)

    class_reps.append(classification_report(y_test_fold, y_pred, zero_division=np.nan))
    conf_mxs.append(confusion_matrix(y_test_fold, y_pred))

avg_train_score = np.mean(train_scores)
avg_test_score = np.mean(test_scores)

print("Times:")
print(times)

print("Train set accuracy:")
print(train_scores)
print("Avg train score:", avg_train_score)

print("Test set accuracy:")
print(test_scores)
print("Avg test score:", avg_test_score)

for i in range(len(class_reps)):
    print(f"No. {i+1}")
    print("······ Clasification Report ······")
    print(class_reps[i])
    print("······ Confusion Matrix  ······")
    print(conf_mxs[i])