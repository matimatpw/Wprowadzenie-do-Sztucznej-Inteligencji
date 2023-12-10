from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 

# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 

print(pd.Series(y.iloc[0])['quality'])
# if pd.Series(y.iloc[0])['quality'] == 3:
#     pd.Series(y.iloc[0])['quality'] = -1
# print(pd.Series(y.iloc[0])['quality'])


for idx in range(len(y)):
    if pd.Series(y.iloc[idx])['quality'] >= 6:
        pd.Series(y.iloc[idx])['quality'] = 1
    else:    
        pd.Series(y.iloc[idx])['quality'] = -1

a,b,c,d,e,f,g,h,i,j,k = (0 for _ in range(11))
for x in range(len(y)):
    if pd.Series(y.iloc[x])["quality"] == -1:
        a +=1
    elif pd.Series(y.iloc[x])["quality"] == 1:
        b += 1
    
print(a)
print(b)
print(a+b == X.shape[0])







    







# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
# metadata 
# print(wine_quality.metadata) 

  
# # variable information 
# print(wine_quality.variables) 
