'''
Using the CIFAR 10 dataset to train a CNN
'''

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Flatten
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import h5py


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


file= './cifar-10-python/cifar-10-batches-py/data_batch_3'

batch=unpickle(file)

category=['airplane','automobile', 'bird','cat','deer','dog','frog','horse',
          'ship','truck']
code=['0','1','2','3','4','5','6','7','8','9']
labels = dict(zip(code, category))

# Read dict-like data
x=batch[b'data']
y=batch[b'labels']


#Train-Test-Split
Xtrain, Xtest, ytrain, ytest = train_test_split(x,y)
ytrain_binary = to_categorical(ytrain)
ytest_binary = to_categorical(ytest)

#reshape data
Xtrain=Xtrain.reshape(7500,3,32,32).transpose(0,2,3,1)
Xtest=Xtest.reshape(2500,3,32,32).transpose(0,2,3,1)

#for i in range(25):
#    plt.subplot(5, 5, i+1)
#    plt.imshow(Xtrain[i])
#    plt.axis('off')

#plt.show()

#Model
model = Sequential()
epochs=12
batchsize=500
model.add(Conv2D(32,kernel_size=(3,3), 
                 activation = 'relu', 
                 input_shape=(32,32,3), 
                 kernel_initializer='glorot_uniform'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu')) #Hidden layer2
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu')) #Hidden layer3
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Conv2D(256, kernel_size=(1,1), activation='relu')) #Output layer
model.add(MaxPooling2D())
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(10,activation = 'softmax'))
model.compile(optimizer='adam',
          loss='binary_crossentropy', #not categorical!!!
          metrics=['accuracy'])

hist = model.fit(Xtrain, ytrain_binary, epochs=epochs, batch_size= batchsize,
                 validation_split = 0.2)

#Save model
model.save('modeldata_1.hdf5')
model.save_weights('CNN_CIFAR10_1.hdf5')

#Evaluate model
model.evaluate(Xtest, ytest_binary)

#plot model
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(range(epochs),hist.history['acc'], label='acc')
ax1.plot(range(epochs),hist.history['val_acc'], label='val_acc')
ax1.legend()
ax2.plot(range(epochs),hist.history['loss'], label='loss')
ax2.plot(range(epochs),hist.history['val_loss'], label='val_loss')
ax2.legend()
plt.show()