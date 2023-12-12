from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np

class PrepareData:
    def __init__(self) -> None:
        self.wine_quality = fetch_ucirepo(id=186) 
        self.X = self.wine_quality.data.features 
        self.y = self.wine_quality.data.targets 
        self.binarize()

    def binarize(self):
        def binarize_label(quality):
            return 1 if quality >= 6 else -1

        self.y['quality'] = self.y['quality'].apply(binarize_label)

    def train_test_split_custom(self, test_size=0.2, random_state=42):
        np.random.seed(random_state)

        num_samples = self.X.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        test_size = int(num_samples * test_size)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]

        X_train, X_test = self.X.iloc[train_indices], self.X.iloc[test_indices]
        y_train, y_test = self.y.iloc[train_indices], self.y.iloc[test_indices]

        return X_train, X_test, y_train, y_test


class SVM:
    def __init__(self, C=1.0, kernel='linear', sigma=0.1, learning_rate=0.01, epochs=1000):
        if kernel == 'linear':
            self.kernel = self.linear_kernel
            self.alpha = None
        elif kernel == 'rbf':
            self.kernel = self.rbf_kernel
            self.sigma = sigma
        else:
            raise ValueError("Kernel type not supported")
        self.C = C
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.b = 0
        self.X = None
        self.y = None

    def linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def rbf_kernel(self, X1, X2):
        return np.exp(-(1 / 2 * self.sigma ** 2) * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) ** 2)

    def fit(self, X_fit, y_fit):
        self.X = X_fit
        self.y = y_fit

        self.alpha = np.random.random(X_fit.shape[0])
        ones_vector = np.ones(X_fit.shape[0])
        yk_sum = np.outer(y_fit, y_fit) * self.kernel(X_fit, X_fit)

        for _ in range(self.epochs):
            gradient = ones_vector - np.sum(yk_sum * self.alpha)

            self.alpha += self.learning_rate * gradient
            self.alpha[self.alpha > self.C] = self.C
            self.alpha[self.alpha < 0] = 0

        index = self.get_index()
        b_fit = y_fit[index] - (self.alpha * y_fit).dot(self.kernel(X_fit, X_fit[index]))
        self.b = np.mean(b_fit)

    def get_index(self):
        return np.where((0 < self.alpha) & (self.alpha < self.C))[0]

    def decision_function(self, X):
        return (self.alpha * self.y).dot(self.kernel(self.X, X)) + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))

def main():
    data = PrepareData()
    print(data.X)

    print(data.y)

main()

    