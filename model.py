from sklearn.decomposition import PCA
from tensorflow.keras.layers import Dense, Flatten, Dropout, ZeroPadding3D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow as tf

from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import losses_utils


def save():
    mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    return mcp_save


def reduce(x_train, x_test):
    pca = PCA(n_components=1024, whiten=True)
    pca.fit(x_train)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    return x_train, x_test


def lstm_cross_entropy(input, number_class):
    model = Sequential()
    model.add(
        LSTM(2048, return_sequences=False, input_shape=input, dropout=0.5)
    )
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_class, activation='softmax'))
    optimizer = Adam(lr=1e-5, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return model



def lstm_huber(input, number_class, delta):
    model = Sequential()
    model.add(
        LSTM(2048, return_sequences=False, input_shape=input, dropout=0.5)
    )
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_class, activation='softmax'))
    optimizer = Adam(lr=1e-5, decay=1e-6)
    huber = tf.keras.losses.Huber(
        delta=delta, reduction=losses_utils.ReductionV2.AUTO, name='huber_loss'
    )
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    return model
