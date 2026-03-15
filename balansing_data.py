from utils import prepare_data
from CustomLogisticRegressionL1 import CustomLogisticRegressionL1
from CustomLogisticRegressionL2 import CustomLogisticRegressionL2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks, RandomUnderSampler

X, y = prepare_data(more_data=True)
print(X.shape)

# smote = SMOTE(random_state=39)
# X, y = smote.fit_resample(X, y)  

# rus = RandomUnderSampler(random_state=39)
# X, y = rus.fit_resample(X, y)

tl = TomekLinks(sampling_strategy='majority')
X, y = tl.fit_resample(X, y)

print(X.shape)

degrees = [2]

avg_train_losses = {d: [] for d in degrees}
avg_test_losses = {d: [] for d in degrees}
avg_train_scores = {d: 0.0 for d in degrees}
avg_test_scores = {d: 0.0 for d in degrees}
times = {d: [] for d in degrees}

for d in degrees:
    print(f"Training for degree {d}")

    poly = PolynomialFeatures(degree=d, include_bias=False)
    X_poly = poly.fit_transform(X)

    scaler = StandardScaler()
    X_poly = scaler.fit_transform(X_poly)

    # print(f"X shape - {X_poly.shape}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=39)

    fold_train_losses = []
    fold_test_losses = []

    fold_train_scores = []
    fold_test_scores = []

    # weights = []

    print("preparation finished")

    for train_index, test_index in skf.split(X_poly, y):
        print("dividing data")
        X_train_fold, X_test_fold = X_poly[train_index], X_poly[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        print("creating and trainig model")

        model = CustomLogisticRegressionL1(lr=0.0001, epochs=500, lambda_L1=0.0001)
        #model = CustomLogisticRegressionL2(lr=0.0001, epochs=500, lambda_L2=0.1)

        start_time = time()
        model.fit(X_train_fold, y_train_fold, X_test_fold, y_test_fold)
        end_time = time()

        print("end of training")

        # print(f"W shape - {model.W.shape}")
        # weights.append(model.W)

        print("\n······ Clasification Report ······")
        print(classification_report(y_test_fold, model.predict(X_test_fold), zero_division=np.nan))
        print("\n······ Confusion Matrix ······")
        print(confusion_matrix(y_test_fold, model.predict(X_test_fold)))
        
        fold_train_losses.append(model.train_losses)
        fold_test_losses.append(model.test_losses)

        fold_train_scores.append(model.score(X_train_fold, y_train_fold))
        fold_test_scores.append(model.score(X_test_fold, y_test_fold))

        times[d].append(end_time - start_time)

    print("calculating avgs")

    fold_train_losses = list(zip(*fold_train_losses))  
    fold_test_losses = list(zip(*fold_test_losses))  
    
    avg_train_losses[d] = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in fold_train_losses]
    avg_test_losses[d] = [sum(epoch_losses) / len(epoch_losses) for epoch_losses in fold_test_losses]

    # avg_train_losses[d] = np.mean(fold_train_losses)
    # avg_test_losses[d] = np.mean(fold_test_losses)

    avg_train_scores[d] = np.mean(fold_train_scores)
    avg_test_scores[d] = np.mean(fold_test_scores)

print("Train set accuracy:")
for key, val in avg_train_scores.items():
    print(f"Degree {key} - {val}")

print("Test set accuracy:")
for key, val in avg_test_scores.items():
    print(f"Degree {key} - {val}")

print("Train set last loss:")
for key, val in avg_train_losses.items():
    print(f"Degree {key} - {val[-1]}")

print("Test set last loss:")
for key, val in avg_test_losses.items():
    print(f"Degree {key} - {val[-1]}")

# print("Weights")
# for w in weights:
#     print(w)


# with open("xxx.txt", "a") as f:
#     f.write("Times:\n")
#     #print("\n\nTimes:")
#     for key, val in times.items():
#         f.write(f"\nDegree {key} - \n")
#         for elem in val:
#             f.write(f"{elem} ")
#         # print(f"Degree {key} - {val}")
#         f.write("\n")

#     f.write("\n\nTrain set losses:\n")
#     # print("\n\nTrain set losses:")
#     for key, val in avg_train_losses.items():
#         f.write(f"\nDegree {key}\n")
#         # print(f"Degree {key}")
#         for elem in val:
#             f.write(f"{elem}\n")
#         # print(val)

#     f.write("\n\nTest set losses:\n")
#     # print("\n\nTest set losses:")
#     for key, val in avg_test_losses.items():
#         f.write(f"\nDegree {key}\n")
#         # print(f"Degree {key}")
#         for elem in val:
#             f.write(f"{elem}\n")
#         # print(val)

for d in degrees:
    plt.plot(avg_train_losses[d], label=f"Train - Degree {d}", color=f"C{d}")
    plt.plot(avg_test_losses[d], linestyle=":", label=f"Test - Degree {d}", color=f"C{d}")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss for data with more features")
plt.legend()
plt.grid(True)
plt.show()