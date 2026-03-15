from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from utils import prepare_data
from time import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

X, y = prepare_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier(max_depth=3)
clf3 = SVC(probability=True, kernel='rbf', gamma=1)
clf4 = KNeighborsClassifier(metric='euclidean', n_neighbors=9)

eclf = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3), ('knn', clf4)],
    voting='soft'
)

estimators = [('LogisticRegression', clf1), 
              ('DecisionTreeClassifier', clf2), 
              ('SVC', clf3), 
              ('KNeighborsClassifier', clf4)]

final_estimator = LogisticRegression()

sclf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator
)

clfs = [('LogisticRegression', clf1), ('DecisionTreeClassifier', clf2), ('SVC', clf3), ('KNeighborsClassifier', clf4), ('VotingClassifier', eclf), ('StackingClassifier', sclf)]

print("preparation finished")

start_time = time()
sclf.fit(X_train, y_train)
end_time = time()

print(f"Time: {end_time-start_time}")

y_pred = sclf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("······ Clasification Report ······")
print(classification_report(y_test, y_pred, zero_division=np.nan))
print("······ Confusion Matrix  ······")
print(confusion_matrix(y_test, y_pred))

# for elem in clfs:
#     name, clf = elem
#     print(name)

#     start_time = time()
#     clf.fit(X_train, y_train)
#     end_time = time()

#     print(f"Time: {end_time-start_time}")

#     y_pred = clf.predict(X_test)
#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print("······ Clasification Report ······")
#     print(classification_report(y_test, y_pred, zero_division=np.nan))
#     print("······ Confusion Matrix  ······")
#     print(confusion_matrix(y_test, y_pred))