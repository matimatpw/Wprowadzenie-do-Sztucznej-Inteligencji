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
        for idx in range(len(self.y)):
            if pd.Series(self.y.iloc[idx])['quality'] >= 6:
                pd.Series(self.y.iloc[idx])['quality'] = 1
            else:    
                pd.Series(self.y.iloc[idx])['quality'] = -1

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




def main():
    data = PrepareData()
    X_train, X_test, y_train, y_test = data.train_test_split_custom(0.1, 42)

    