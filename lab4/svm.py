from enum import Enum
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from ucimlrepo import fetch_ucirepo
import pandas as pd

class kernel_types (Enum):
    LINEAR = 0
    RBF = 1

class optim_params:
    def __init__(self,C:float=1.0,kernel:kernel_types=kernel_types.LINEAR,lr:float=0.01,iters:int=300, sigma:float=0.2) -> None:
        self.C = C
        self.kernel = kernel
        self.learning_rate = lr
        self.iters = iters
        self.sigma = sigma
        self.zero = 0
        self.half = 0.5


class SVM:
    def __init__(self, optim_param:optim_params) -> None:
        self.params = optim_param
        if self.params.kernel == kernel_types.LINEAR:
            self.kernel = self.linear_kernel_func
            print("linear")
        elif self.params.kernel == kernel_types.RBF:
            print("rbf")
            self.kernel = self.rbf_kernel_func
            self.sigma = self.params.sigma
        else: raise ValueError("Kernel not supported")
        self.C = self.params.C
        self.alpha = None
        self.step_size = self.params.learning_rate
        self.iters = self.params.iters
        self.X = None
        self.y = None
        self.loses = []

    def linear_kernel_func(self, X1:np.array, X2:np.array) -> np.array:
        return np.dot(X1, X2.T) + 1

    def rbf_kernel_func(self, X1:np.array, X2:np.array) -> np.array:
        return np.exp(-(1 / self.sigma ** 2) * np.linalg.norm(X1[:, np.newaxis] - X2[np.newaxis, :], axis=2) ** 2)

    def update_loss(self,my_yk_sum: np.array) -> None:
        loss = np.sum(self.alpha) - self.params.half * np.sum(np.outer(self.alpha, self.alpha) * my_yk_sum)
        self.loses.append(loss)

    def update_alpha(self,gradient: np.array) -> None:
        self.alpha += self.step_size * gradient
        self.alpha[self.alpha > self.C] = self.C
        self.alpha[self.alpha < 0] = self.params.zero

    def get_bias(self,y_targets, X_features) -> float:
        idx = np.where((self.alpha > 0 ) & (self.alpha < self.C))[0]
        if len(idx) == 0:
            return 0
        else:
            b_fit = y_targets[idx] - (self.alpha * y_targets).dot(self.kernel(X_features, X_features[idx]))
            return np.mean(b_fit)

    def fit(self, X_features, y_targets):
        self.X = X_features
        self.y = y_targets

        self.alpha = np.random.random(X_features.shape[0])
        yk_sum = np.outer(y_targets, y_targets) * self.kernel(X_features, X_features)

        for _ in range(self.iters):
            gradient = np.ones(X_features.shape[0]) - yk_sum.dot(self.alpha)
            self.update_alpha(gradient)

            self.update_loss(yk_sum)

        self.bias = self.get_bias(y_targets, X_features)

    def predict(self, X:np.array) -> np.array:
        return np.sign(self.decision_function(X))

    def decision_function(self, X:np.array) -> np.array:
        return (self.alpha * self.y).dot(self.kernel(self.X, X)) + self.bias



class PrepareData:
    def __init__(self) -> None:
        self.wine_quality = fetch_ucirepo(id=186)
        self.X = self.wine_quality.data.features
        self.y = self.wine_quality.data.targets
        self.binarize()

    def binarize(self) -> None:
        for idx in range(len(self.y)):
            if pd.Series(self.y.iloc[idx])['quality'] >= 6:
                pd.Series(self.y.iloc[idx])['quality'] = 1
            else:
                pd.Series(self.y.iloc[idx])['quality'] = -1


    def train_test_split_custom(self, test_size=0.2, random_state=18) -> tuple:
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

    def wrap_targets(self, y_to_train, y_to_test) -> tuple:
        y_to_train = y_to_train.ravel()
        y_to_test = y_to_test.ravel()
        return y_to_train, y_to_test

if __name__ == '__main__':
    data = PrepareData()
    
    X_train, X_test, y_train, y_test = train_test_split(np.array(data.X),np.array(data.y), test_size=0.5, random_state=18)
    op = optim_params(C=1.0, kernel=kernel_types.RBF,lr=1e-4, iters=10)


    linear_svm_model = SVM(op)
    print(y_train.shape, y_test.shape)
    y_train, y_test = data.wrap_targets(y_train,y_test)
    print(y_train.shape, y_test.shape)
    # y_train = y_train.ravel()
    # y_test = y_test.ravel()
    linear_svm_model.fit(X_train, y_train)

    test_pred =       linear_svm_model.predict(X_test)
    train_pred  =       linear_svm_model.predict(X_train)

    test_score =  accuracy_score(y_test, test_pred)
    train_score = accuracy_score(y_train, train_pred)
    print("test accuracy: " , test_score)
    print("train accuracy: ", train_score)
    print("Train/ Test accuracy:(shuld not be very high)", (train_score/test_score))

    # tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
    # print("confusion matrix: ", (tn, fp, fn, tp))

    # print(linear_svm_model.loses)
    # plt.plot(linear_svm_model.loses)
    # plt.show()
