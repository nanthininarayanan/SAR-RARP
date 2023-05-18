import matplotlib
matplotlib.use('AGG')

from keras.datasets import cifar10
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPool2D, Input, ZeroPadding2D,TimeDistributed,LSTM)
from keras.models import Sequential

def TDCNNLSTM():
#define the model

    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'), input_shape=(16, 112, 112, 3)))
    model.add(TimeDistributed(Conv2D(32, (3,3), kernel_initializer="he_normal", activation='relu')))
    model.add(TimeDistributed(MaxPool2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPool2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(128, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPool2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(256, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPool2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(Conv2D(512, (3,3), padding='same', activation='relu')))
    model.add(TimeDistributed(MaxPool2D((2, 2), strides=(2, 2))))

    model.add(TimeDistributed(Flatten()))

    model.add(Dropout(0.5))
    model.add(LSTM(512, return_sequences=False, dropout=0.5))
    model.add(Dense(8, activation='softmax'))
    
    return model