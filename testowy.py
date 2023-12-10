from ucimlrepo import fetch_ucirepo 
import pandas as pd
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 

# data (as pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 

a,b,c,d,e,f,g,h,i,j,k,l = (0 for i in range(12))

# for x in range(y.shape[0]):
#     if pd.Series(y.iloc[x])["quality"] == 0:
#         a +=1
#     elif pd.Series(y.iloc[x])["quality"] == 1:
#         b += 1
#     elif pd.Series(y.iloc[x])["quality"] == 2:
#         c += 1
#     elif pd.Series(y.iloc[x])["quality"] == 3:
#         d += 1
#     elif pd.Series(y.iloc[x])["quality"] == 4:
#         e += 1
#     elif pd.Series(y.iloc[x])["quality"] == 5:
#         f += 1
#     elif pd.Series(y.iloc[x])["quality"] == 6:
#         g += 1
#     elif pd.Series(y.iloc[x])["quality"] == 7:
#         h += 1
#     elif pd.Series(y.iloc[x])["quality"] == 8:
#         i += 1
#     elif pd.Series(y.iloc[x])["quality"] == 9:
#         j += 1
#     elif pd.Series(y.iloc[x])["quality"] == 10:
#         k += 1
#     elif pd.Series(y.iloc[x])["quality"] == 11:
#         l += 1



print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
print(h)
print(i)
print(j)
print(k)
print(l)







  
# metadata 
# print(wine_quality.metadata) 

  
# # variable information 
# print(wine_quality.variables) 
