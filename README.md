# Simple-ML-Implementation
Simple Machine learning task/s 
## Running the Program
Good Day! You should be able to run the program with ease once you have install the packages listed in the **requirements.txt** file
## What's going on in This CODE:
### Pandas
Used to read data in the .csv file and store in it a format so that python can be used to analysis/ process it  
### sklearn.neighbors and KNeighborsClassifier
contains functionality relating to K-Nearest Neighbors 
### sklearn.model_selection iand train_test_split
contains functionality to split data into groups to train and test the model
### sklearn.metrics and accuracy_score
contains functionality to calculate the accuracy of the model
### sklearn.preprocessing and StandardScaler
contains functionality for standardizing the data
### X and  Y
`X = dataframe[["Length","Width","LengthGroove"]]`  
**X** consists of the (input)independent variables  
Assigns a dataframe to **X** which consists of the columns "Length","Width" and "LengthGroove"
`Y = dataframe["Class"]`  
**Y** consists of the dependent variables  
assigned a dataframe consisting of the column "Class"  
`x_scaled= scaler.fit_transform(X)`  
**x_scaled** is the scaled verion of the (input)independent variables to be used for learning.
### train_test_split
`X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2 )`
Splits the data into 4 variables , 2 pairs, one pair for training the model and the other pair for testing the model. 80% of the data set will be used for training while 20 % will be used for testing.
### KNN model 
K-Nearest Neighbours is a Machine learning algorithm mostly used in classifaction problems, its requires normalization/standardization of variables. The algorithm assigns classification based on the nears data points i.e. looks at the nearests values and decides the current value based on those around it.  
`knn_model = KNeighborsClassifier(n_neighbors=5)`  
makes algorithm look at 5 of the nearesst values  
`knn_model.fit(X_train,Y_train)`
trains model with the training data  
`predictions= knn_model.predict(X_test)`
tests model by making predictions with test data  
`score = accuracy_score(Y_test,predictions)`  
calculates the accuracy of the predictions by comparing the predictions to the test data  
## Resources used (Well that I can remember)
[Python Machine Learning Tutorial](https://www.youtube.com/watch?v=7eh4d6sabA0&t=2207s)  
[Machine Learning for Everybody – Full Course](https://www.youtube.com/watch?v=i_LwzRVP7bg&t=2423s) 
[Visual guide to learning Data Science (Image)](https://towardsdatascience.com/how-to-build-a-machine-learning-model-439ab8fb3fb1)
[Building a Machine Learning model in Python](https://www.youtube.com/watch?v=29ZQ3TDGgRQ&t=892s)
[No.9 Wheat Seeds Dataset](https://machinelearningmastery.com/standard-machine-learning-datasets/)  