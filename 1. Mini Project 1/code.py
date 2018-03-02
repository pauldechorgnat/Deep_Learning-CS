#%% Importing packages
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils

import data_generation


#%%
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD, Adam

#%% Defining training and test data

[X_train, Y_train] = generate_dataset_classification(300, 20)
[X_test, Y_test] = generate_test_set_classification()

Y_train_cat = np_utils.to_categorical(Y_train)

#%%
# clearing keras backend 
K.clear_session()

# function to generate simple linear models
def generate_linear_model(X_train = X_train, nb_neurons = 40, opt = 'adam'):
    model = Sequential()
    model.add(Dense(nb_neurons, input_shape = (X_train.shape[1],), activation = 'relu'))
    model.add(Dense(3, activation = 'softmax'))
    model.summary()
    
    model.compile(loss = 'categorical_crossentropy', metrics = ['accuracy'], optimizer=opt)
    
    return model

# Defining optimizers
adam = Adam(lr = .00001)
sgd = SGD(lr = .0001)

nb_neurons = 20


linear_model_adam = generate_linear_model(opt = adam, nb_neurons= 20)
linear_model_sgd = generate_linear_model(opt = "sgd", nb_neurons = 20)


#%% Fitting the training data 
#
linear_model_sgd.fit(X_train, Y_train_cat, epochs = 100,
                     batch_size = 32, verbose = 1)

linear_model_adam.fit(X_train, Y_train_cat, epochs = 100, 
                      batch_size = 32, verbose = 1)

#%% Printing resuslts
predictions_adam = np_utils.to_categorical(linear_model_adam.predict_classes(X_test))
predictions_sgd = np_utils.to_categorical(linear_model_sgd.predict_classes(X_test))

from sklearn.metrics import classification_report

print("\nADAM Optimizer", "\n",classification_report(Y_test, predictions_adam))
print("\nSGD  Optimizer", "\n",classification_report(Y_test, predictions_sgd))
#%%

weights1_adam = np.array(linear_model_adam.get_weights()[0]).reshape(72,72,nb_neurons)
weights2_adam = np.array(linear_model_adam.get_weights()[2])
total_weights_adam = np.dot(weights1_adam, weights2_adam)

weights1_sgd = np.array(linear_model_sgd.get_weights()[0]).reshape(72,72,nb_neurons)
weights2_sgd = np.array(linear_model_sgd.get_weights()[2])
total_weights_sgd = np.dot(weights1_sgd, weights2_sgd)

#%%
print('ADAM Optimizers')
for i in range(3):
    plt.imshow(total_weights_adam[:,:,i],cmap = 'hot')
    plt.show()
print('SGD Optimizers')
for i in range(3):
    plt.imshow(total_weights_sgd[:,:,i],cmap = 'hot')
    plt.show()

#%% Shapes are allowed to move
[X_train, Y_train] = generate_dataset_classification(300, 20, True)
[X_test, Y_test] = generate_test_set_classification()
#%%
# clearing keras backend 
K.clear_session()
#%% Defining optimizers
adam = Adam(lr = .0001)
sgd = SGD(lr = .0001)
#defining a linear classification model 
linear_model = Sequential()

# adding layers
linear_model.add(Dense(20, input_shape =(X_train.shape[1],), activation = 'relu'))
#linear_model.add(Dense(10, activation = "relu"))
linear_model.add(Dense(3, activation = "softmax"))


linear_model.summary()

linear_model_sgd = linear_model
linear_model_adam = linear_model

#%% Compiling with SGD and Adam optimizer
linear_model_sgd.compile(optimizer=sgd, loss="categorical_crossentropy", metrics = ["accuracy"])
linear_model_adam.compile(optimizer=adam, loss="categorical_crossentropy", metrics = ["accuracy"])
#%%


Y_train_cat = np_utils.to_categorical(Y_train)
Y_test_cat = np_utils.to_categorical(Y_test)

linear_model_sgd.fit(X_train, Y_train_cat, validation_split=.2, epochs = 20,
                     batch_size = 32, verbose = 1)

linear_model_adam.fit(X_train, Y_train_cat, validation_split=.2, epochs = 20, 
                      batch_size = 32, verbose = 1)

#%% Evaluating performances

print('\nADAM Optimizer\n', linear_model_adam.evaluate(X_test, Y_test))
print('\nSGD  Optimizer\n', linear_model_sgd.evaluate(X_test, Y_test))

#%% Building a Convolutionnal Neural Net
K.clear_session()

X_train_cnn = X_train.reshape(300,72,72)

cnn_model = Sequential()

cnn_model.add(Conv2D(16, kernel_size=(5,5), input_shape = (72,72,)))
#%%
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Dropout(.25))
cnn_model.add(Flatten())
cnn_model.add(Dense(3, activation = 'softmax'))
#%%
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#%%
cnn_model.fit(X_train_cnn, Y_train_cat, epochs = 10, batch_size=32)
