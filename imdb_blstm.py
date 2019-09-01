import numpy as np
import os, pickle, glob, collections
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras import optimizers
from nltk.tokenize import word_tokenize

max_words = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 256
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
# pre tokenized data
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Pad sequences (samples x time)')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

y_train = np.array(y_train)
y_test = np.array(y_test)
model = Sequential()
model.add(Embedding(max_words, 200, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=1,
          validation_data=[x_test, y_test])

# loss: 0.4194 - acc: 0.8106 - val_loss: 0.4010 - val_acc: 0.8235
