# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 18:51:41 2023

@author: sara
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout



path = 'C:/Users/sara/Desktop/ML&DM Programming/ML&DM_Chapter5_film3/'
data = pd.read_csv(path + "data.csv")

Information_of_data = data.info
print(Information_of_data)

X = data.iloc[:, 2:-1].values
y = data.iloc[:, 1].values
labelencoder_X_1 = LabelEncoder()#baraye tabdil huruf be adad
y = labelencoder_X_1.fit_transform(y)#emale label encoder ruye y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()#normalize
X_train = sc.fit_transform(X_train)#emale normalize ruye X_train 
X_test = sc.transform(X_test)#emale normalize ruye X_test


NN = Sequential()
#NN.add(Dense(13, activation='relu', input_shape=(29,)))#dar avalin layer inputshape ra ke tedade feachers hast gharar midim
#NN.add(Dropout(0.1))
NN.add(Dense(1, activation='sigmoid'), input_shape=(29,))#activation chon layer akhar chon output ya 0 ya 1 ast sigmoid gozashtim
NN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])#chon 2 class hast loss ro binary_crossentropy gereftim
NN.fit(X_train, y_train, batch_size = 32, epochs = 150)

val_loss, val_acc = NN.evaluate(X_test, y_test)