# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 22:57:00 2019

@author: megan
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix,precision_score
from sklearn.model_selection import train_test_split

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
red = pd.read_csv("winequality-red.csv", 
                   delimiter = ";", header=0)
#pd.isnull(red)
#print(red.head())

X = red[:,0:11]   
y = red[:,11]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Initialize the constructor
model = Sequential()

# Add an input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))
# Add one hidden layer 
model.add(Dense(8, activation='relu'))
# Add an output layer 
model.add(Dense(1, activation='sigmoid'))
print(model.summary())
print(model.output_shape)
print(model.get_config())

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
model.fit(X_train, y_train,epochs=20, batch_size=1, verbose=1)

#predict values
y_pred = model.predict(X_test)
print(y_pred[:5])
print(y_test[:5])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Evaluate Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
score = model.evaluate(X_test, y_test,verbose=1)
print(score)

confusion_matrix(y_test, y_pred)
precision_score(y_test, y_pred)