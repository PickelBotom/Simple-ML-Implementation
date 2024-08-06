#!/usr/bin/env python3

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
dataframe = pd.read_csv("seeds_DS.csv")

X = dataframe[["Length","Width","LengthGroove"]]
Y = dataframe["Class"]

scaler= StandardScaler()
x_scaled= scaler.fit_transform(X)  

X_train,X_test,Y_train,Y_test=train_test_split(x_scaled,Y,test_size=0.2 )

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train,Y_train)

predictions= knn_model.predict(X_test)

score = accuracy_score(Y_test,predictions)
print("This model has an accuracy of roughly {:.2f}%".format(score*100)) 

