from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class MixtureOfExpertsSoft(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, n_experts=3, gater=None, expert_params=None):
        self.n_experts = n_experts
        self.base_model = base_model
        self.expert_params = expert_params or {}
        self.experts = [base_model(**self.expert_params) for _ in range(n_experts)]
        self.gater = gater or LogisticRegression(multi_class='multinomial')

    def fit(self, X, y):
        assignments = np.random.randint(0, self.n_experts, size=len(X))

        for i in range(self.n_experts):
            X_i = X[assignments == i]
            y_i = y[assignments == i]
            self.experts[i].fit(X_i, y_i)

        self.gater.fit(X, assignments)

        return self
    
    def predict_proba(self, X):
        expert_weights = self.gater.predict_proba(X)
        n_classes = self.experts[0].predict_proba(X[:1]).shape[1]
        probs = np.zeros((X.shape[0], n_classes))

        for idx, expert in enumerate(self.experts):
            expert_probs = expert.predict_proba(X)
            weights = expert_weights[:, idx].reshape(-1, 1)
            probs += weights * expert_probs

        return probs
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
class MixtureOfExpertsSoftV2(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, n_experts=3, gater=None, expert_param_list=None):
        self.n_experts = n_experts
        self.base_model = base_model
        self.gater = gater or LogisticRegression(multi_class='multinomial')

        if expert_param_list is None:
            expert_param_list = [
                {'lr': 0.001, 'epochs': 100, 'lambda_L1': 0.0001},
                {'lr': 0.01, 'epochs': 500, 'lambda_L1': 0.001},
                {'lr': 0.0001, 'epochs': 1000, 'lambda_L1': 0.0}
            ]

        self.experts = [base_model(**params) for params in expert_param_list]

    def fit(self, X, y):
        assignments = np.random.randint(0, self.n_experts, size=len(X))

        for expert in self.experts:
            expert.fit(X, y)

        self.gater.fit(X, assignments)

        return self
    
    def predict_proba(self, X):
        expert_preds = np.array([expert.predict_proba(X) for expert in self.experts])
        gater_weights = self.gater.predict_proba(X)
        weighted_preds = np.einsum('ens,se->sn', expert_preds, gater_weights)

        return weighted_preds
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
    
class MixtureOfExpertsHard(BaseEstimator, ClassifierMixin):
    def __init__(self, base_model, n_experts=3, gater=None, expert_params=None):
        self.n_experts = n_experts
        self.base_model = base_model
        self.expert_params = expert_params or {}
        self.experts = [base_model(**self.expert_params) for _ in range(n_experts)]
        self.gater = gater or KNeighborsClassifier(n_neighbors=1)

    def fit(self, X, y):
        assignments = np.random.randint(0, self.n_experts, size=len(X))

        for i in range(self.n_experts):
            X_i = X[assignments == i]
            y_i = y[assignments == i]
            self.experts[i].fit(X_i, y_i)

        self.gater.fit(X, assignments)

        return self
    
    def predict_proba(self, X):
        expert_idxs = self.gater.predict(X)
        n_classes = self.experts[0].predict_proba(X[:1]).shape[1]
        probs = np.zeros((X.shape[0], n_classes))

        for i, idx in enumerate(expert_idxs):
            probs[i] = self.experts[idx].predict_proba(X[i].reshape(1, -1))

        return probs
    
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)