from ucimlrepo import fetch_ucirepo 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 

# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 

for idx in range(len(y)):
    if pd.Series(y.iloc[idx])['quality'] >= 6:
        pd.Series(y.iloc[idx])['quality'] = 1
    else:    
        pd.Series(y.iloc[idx])['quality'] = -1

def train_test_split_custom(X, y, test_size=0.2, random_state=42):

    np.random.seed(random_state)

    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    test_size = int(num_samples * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

    return X_train, X_test, y_train, y_test


print(y)

# X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.2, random_state=42)
    
# print(X_train.iloc[0])
# print(X_test.iloc[0])







  
# metadata 
# print(wine_quality.metadata) 

  
# # variable information 
# print(wine_quality.variables) 
