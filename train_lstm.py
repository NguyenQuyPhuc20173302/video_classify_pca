import pickle
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint

import model



with open('x_train.pkl', 'rb') as f:
    x_train = pickle.load(f)
with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)


with open('x_test.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

x_train = x_train.reshape(61607, 1, 2048)
x_test = x_test.reshape(28843, 1, 2048)
model = model.lstm((1, 2048), 101)
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=32, callbacks=[mcp_save])
