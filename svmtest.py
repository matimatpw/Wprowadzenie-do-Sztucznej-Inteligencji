import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from ucimlrepo import fetch_ucirepo 
import pandas as pd

class SVM:
    def __init__(self, C=1.0, kernel='linear', sigma=0.1, learning_rate=0.01, epochs=1000):
        if kernel == 'linear':
            self.kernel = self.linear_kernel
            self.alpha = None
        else:
            self.kernel = self.rbf_kernel
            self.sigma = sigma
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


class PrepareData:
    def __init__(self) -> None:
        self.wine_quality = fetch_ucirepo(id=186) 
        self.X = self.wine_quality.data.features 
        self.y = self.wine_quality.data.targets 
        self.binarize()

    def binarize(self):
        for idx in range(len(self.y)):
            if pd.Series(self.y.iloc[idx])['quality'] >= 6:
                pd.Series(self.y.iloc[idx])['quality'] = 1
            else:    
                pd.Series(self.y.iloc[idx])['quality'] = -1


    def train_test_split_custom(self, test_size=0.2, random_state=18):
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


if __name__ == '__main__':
    data = PrepareData() 
    X_train, X_test, y_train, y_test = train_test_split(np.array(data.X),np.array(data.y), test_size=0.2, random_state=18)


    linear_svm_model = SVM(C=0.1, learning_rate=1e-10, epochs=300, kernel='rbf', sigma=0.5)
    
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    linear_svm_model.fit(X_train, y_train)

    predictions = linear_svm_model.predict(X_test)
    print("Y TEST:",y_test.shape)
    print("PREDICTION:" , predictions.shape)

    print("accuracy:", accuracy_score(y_test, predictions))
    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    print("confusion matrix: ", (tn, fp, fn, tp))



    