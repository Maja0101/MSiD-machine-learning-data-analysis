from utils import prepare_data
from sklearn.model_selection import StratifiedKFold
import numpy as np
from CustomLogisticRegression import CustomLogisticRegression
from time import time

X, y = prepare_data()

model = CustomLogisticRegression()

lst_accu_stratified_train = []
lst_accu_stratified = []
times = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=39)

for train_index, test_index in skf.split(X, y):
	print("dividing data")
	X_train_fold, X_test_fold = X[train_index], X[test_index]
	y_train_fold, y_test_fold = y[train_index], y[test_index]

	print("creating and trainig model")
	start_time = time()
	model.fit(X_train_fold, y_train_fold)
	end_time = time()

	times.append(end_time - start_time)

	lst_accu_stratified_train.append(model.score(X_train_fold, y_train_fold))
	lst_accu_stratified.append(model.score(X_test_fold, y_test_fold))

print("Times:")
print(times)

print('\nTrain accuracy per fold:')
print(lst_accu_stratified_train)
print('Maximum Accuracy:',
	max(lst_accu_stratified_train))
print('Minimum Accuracy:',
	min(lst_accu_stratified_train))
print('Overall Accuracy:',
	np.mean(lst_accu_stratified_train))
print('Standard Deviation:', np.std(lst_accu_stratified_train))

print('\nAccuracy per fold:')
print(lst_accu_stratified)
print('Maximum Accuracy:',
	max(lst_accu_stratified))
print('Minimum Accuracy:',
	min(lst_accu_stratified))
print('Overall Accuracy:',
	np.mean(lst_accu_stratified))
print('Standard Deviation:', np.std(lst_accu_stratified))
