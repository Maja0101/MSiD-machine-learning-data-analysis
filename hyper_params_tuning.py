from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from utils import prepare_data
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from time import time
import seaborn as sns
import matplotlib.pyplot as plt

X, y = prepare_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

# param_grid_svc = {
#     'C': [0.1, 1, 10],
#     'kernel': ['linear', 'rbf', 'poly'],
#     'gamma': [0.001, 0.01, 0.1, 1]
# }

# grid_search = GridSearchCV(SVC(), param_grid_svc, cv=5, scoring='accuracy')

# param_grid_tree = {
#     'max_depth': [3, 5, 10, None],
#     'min_samples_split': [2, 5, 10],
#     'criterion': ['gini', 'entropy']
# }

# grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid_tree, cv=5)

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)

print("GridSearchCV created")

start_time = time()
grid_search.fit(X_train, y_train)
end_time = time()

print(f"Time: {end_time-start_time}")

print("Best score:")
print(grid_search.best_score_)

print("Best parameters:")
print(grid_search.best_params_)

print("Best estimator:")
print(grid_search.best_estimator_)

y_pred = grid_search.best_estimator_.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("······ Clasification Report ······")
print(classification_report(y_test, y_pred, zero_division=np.nan))
print("······ Confusion Matrix  ······")
print(confusion_matrix(y_test, y_pred))

results = pd.DataFrame(grid_search.cv_results_)
print(results)