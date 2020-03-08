import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.optimizers import adam

batch_size = 512
epochs = 20



## load data
N_train = pd.read_csv('app/data/mnist_train.csv')
N_test = pd.read_csv('app/data/mnist_test.csv')

## process data
Y_train = np.array(N_train['5'])
X_train =  np.array(N_train.drop('5',axis = 1)).reshape(N_train.shape[0], 28,28,1)
X_train = X_train.astype('float32')
X_train /= 255
Y_test = np.array(N_test['7'])
X_test =  np.array(N_test.drop('7',axis = 1)).reshape(N_test.shape[0], 28,28,1)
X_test = X_test.astype('float32')
X_test /= 255

x_train, x_validate, y_train, y_validate = train_test_split(X_train, Y_train, test_size = 0.2, random_state = 12345)

## train model

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape = (28,28,1), activation = 'relu'))
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(Conv2D(128,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=adam(lr = 0.001), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, nb_epoch = epochs, verbose=1, validation_data=(x_validate, y_validate ))

score = model.evaluate(X_test, Y_test)
print (score)

## save model

model_json = model.to_json()
with open('model_json','w') as json_file:
	json_file.write(model_json)

#save weights to HDF5
model.save_weights('model.h5')
