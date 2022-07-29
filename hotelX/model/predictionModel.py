# Import all the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics
# import from sklearn to split the data into train and test
from sklearn.model_selection import train_test_split
#import the LinearRegression estimator from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import os 
import glob
def searchFile():
    os.chdir("C:/Users/Google Prep Oct 22/Music/HotelMLProject-master/HotelMLProject-master/hotelX/media/media")
    for file in glob.glob("*.csv"):
        # print(file)
        # fileName = str(file)
        return str(file)

f = searchFile()
fileName = f
df = pd.read_csv(fileName)

class HotelModel:
    def __init__(self):
        pass
    
    def cleanDataset(self):
        df.drop(["date"], axis=1, inplace=True)
        df.drop(["Sno"], axis=1, inplace=True)

    def collectionDataset(self):
        print(df)
        plt.hist(df["Occupancy"])
        plt.xlabel("No of rooms")
        plt.ylabel("Frequency")
        plt.show()

    def randomForest(self):
        print("random Forest : ")
        try: 
            a = HotelModel()
            a.cleanDataset()
            # Split the dataset from X and y
            X = df.iloc[:, :-1] # X = all feature, except target
            y = df.iloc[:,-1] # y = only target
            X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=100)
            model = RandomForestRegressor()
            model.fit(X_train, y_train) # fitted the model using training dataset
            y_pred = model.predict(X_test)
            acc = 100 - np.sqrt(mean_squared_error(y_test, y_pred))
            plt.scatter(y_test, y_pred)
            plt.show()
            return acc
        except:
            X = df.iloc[:, :-1] # X = all feature, except target
            y = df.iloc[:,-1] # y = only target
            X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=100)
            model = RandomForestRegressor()
            model.fit(X_train, y_train) # fitted the model using training dataset
            y_pred = model.predict(X_test)
            acc = 100 - np.sqrt(mean_squared_error(y_test, y_pred))
            plt.scatter(y_test, y_pred)
            plt.show()
            return acc  

    def linearRegression(self):
        print("linear regression : ")
        try:
            a = HotelModel()
            a.cleanDataset()
            
            # df.drop(["date"], axis=1, inplace=True)
            # df.drop(["Sno"], axis=1, inplace=True)
            # Split the dataset from X and y
            X = df.iloc[:, :-1] # X = all feature, except target
            y = df.iloc[:,-1] # y = only target
            X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=100)
            model = LinearRegression()
            model.fit(X_train, y_train)
            #To generate predictions from our model using the predict method
            predictions = model.predict(X_test)
            #to do this is plot the two arrays using a scatterplot. 
            plt.scatter(y_test, predictions)
            # plt.hist(y_test - predictions)
            plt.show()
            sol = metrics.mean_squared_error(y_test, predictions) 
            accuracy = 100-sol
            return accuracy
        except:
            X = df.iloc[:, :-1] # X = all feature, except target
            y = df.iloc[:,-1] # y = only target
            X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=100)
            model = LinearRegression()
            model.fit(X_train, y_train)
            #To generate predictions from our model using the predict method
            predictions = model.predict(X_test)
            #to do this is plot the two arrays using a scatterplot. 
            plt.scatter(y_test, predictions)
            # plt.hist(y_test - predictions)
            plt.show()
            sol = metrics.mean_squared_error(y_test, predictions) 
            accuracy = 100-sol
            return accuracy



    def decisionTree(self):
        print("decision Tree : ")
        try:
            a = HotelModel()
            a.cleanDataset()
            # Split the dataset from X and y
            X = df.iloc[:, :-1] # X = all feature, except target
            y = df.iloc[:,-1] # y = only target
            X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=100)
            regressor = DecisionTreeRegressor()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            plt.scatter(y_test, y_pred)
            plt.show()
            sol = metrics.mean_squared_error(y_test, y_pred) 
            accuracy = 100-sol
            return accuracy
        except:
            X = df.iloc[:, :-1] # X = all feature, except target
            y = df.iloc[:,-1] # y = only target
            X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=100)
            regressor = DecisionTreeRegressor()
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            plt.scatter(y_test, y_pred)
            plt.show()
            sol = metrics.mean_squared_error(y_test, y_pred) 
            accuracy = 100-sol
            return accuracy

# p = HotelModel()
# acc = p.linearRegression()
# acc = p.randomForest()
# acc = p.decisionTree()
# print(acc)