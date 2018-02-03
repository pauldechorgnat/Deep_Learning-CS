# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 12:29:10 2018

@author: Paul
"""

import data_generation
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import numpy as np


#%% Loading data
[X_train, Y_train] = data_generation.generate_dataset_classification(300, 20, True)
X_train = X_train.reshape(-1, 72, 72, 1)
Y_train = np_utils.to_categorical(Y_train)
#%% Defining a classifier
K.clear_session()
classifier = Sequential()

classifier.add(Conv2D(16, kernel_size=(5,5), input_shape = (X_train.shape[1], X_train.shape[2],1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dropout(.25))
classifier.add(Dense(20, activation = 'relu'))
classifier.add(Dense(3, activation = 'softmax'))

#%% Compiling

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, Y_train, epochs = 20, batch_size = 32, verbose = 1)
#%% Generating our model
[X_test, Y_test] = data_generation.generate_test_set_classification()

X_test = X_test.reshape(-1, 72, 72, 1)
#%% Predicting on test data
classification_predictions = np_utils.to_categorical(classifier.predict_classes(X_test))

from sklearn.metrics import classification_report
print('\n', classification_report(Y_test, classification_predictions))
#%% Loafing train data

[X_train, Y_train] = data_generation.generate_dataset_regression(3000, 20)
X_train = X_train.reshape(-1, 72, 72, 1)

#%%

def generate_new_features(Y_train):
    gravity_centres = np.array([np.mean(Y_train[:,[0,2,4]], axis = 1),np.mean(Y_train[:,[1,3,5]], axis = 1)]).T
    
    A = Y_train[:,0:2] - gravity_centres
    B = Y_train[:,2:4] - gravity_centres
    C = Y_train[:,4:6] - gravity_centres
    
    new_data = np.concatenate([A,B,C], axis = 1)
    
    distances_max = np.argmax(new_data[:,[0,2,4]], axis = 1)*2
    distances_min = np.argmin(new_data[:,[0,2,4]], axis = 1)*2
    distances_autres = 6 - distances_max-distances_min
    right_vectors = [[new_data[i,distances_max[i]], new_data[i,distances_max[i]+1]] for i in range(Y_train.shape[0])]
    left_vectors  = [[new_data[i,distances_min[i]], new_data[i,distances_min[i]+1]] for i in range(Y_train.shape[0])]
    other = [[new_data[i,distances_autres[i]], new_data[i,distances_autres[i]+1]] for i in range(Y_train.shape[0])]
    final_data = np.concatenate([gravity_centres, right_vectors, left_vectors], axis = 1).reshape(-1, 6)
    
    return final_data


Y_train_transformed = generate_new_features(Y_train)

def recreate_features(Y_train_transformed):
    
    gravity_centres = Y_train_transformed[:,:2]
    right = Y_train_transformed[:,2:4] + gravity_centres
    left = Y_train_transformed[:,4:6] + gravity_centres
    #third = Y_train_transformed[:,6:8] + gravity_centres
    third =gravity_centres*3 - right -left
    
    return np.concatenate([right, left, third], axis = 1)
#%%
data_generation.visualize_prediction(X_train[0], Y_train[0])
data_generation.visualize_prediction(X_train[0], recreate_features(generate_new_features(Y_train[[0]])))
#%% Dzfininf a regressor
regressor = Sequential()
regressor.add(Conv2D(16, kernel_size=(5,5), input_shape = (X_train.shape[1], X_train.shape[2], 1), activation = 'relu'))
regressor.add(MaxPooling2D(pool_size=(2,2)))
regressor.add(Flatten())
regressor.add(Dropout(.25))
regressor.add(Dense(20, activation = "relu"))
regressor.add(Dense(6))
#%% Compiling regressor

regressor.compile(optimizer = Adam(lr = .0001), loss = 'mse')

regressor.fit(X_train, Y_train_transformed, batch_size = 16, epochs = 100, verbose = 1)

#%%
for i in range(10):
    data_generation.visualize_prediction(X_train[i], recreate_features(regressor.predict(X_train[i].reshape(-1,72,72,1))))

#%%

[X_test, Y_test] = data_generation.generate_test_set_regression()
#%%
Y_test_transfo = generate_new_features(Y_test)

X_test = X_test.reshape(-1, 72, 72, 1)
predictions_regression = recreate_features(regressor.predict(X_test))

for i in range(300):
    data_generation.visualize_prediction(X_test[i], recreate_features(regressor.predict(X_test[i].reshape(-1,72,72,1))))

#%%
regressor.evaluate(X_test, Y_test_transfo)