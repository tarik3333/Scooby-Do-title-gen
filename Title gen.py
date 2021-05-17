# -*- coding: utf-8 -*-
"""
Created on Mon May 17 08:57:04 2021

@author: Tarık Buğra Tufan
"""
import numpy
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

file = open("Scooby_titles.txt").read()
def tokenize_words(input):
    
    input = input.lower()

    
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(input)

   
    return " ".join(tokens)
processed_inputs=tokenize_words(file)
chars = sorted(list(set(processed_inputs)))
char_to_num = dict((c, i) for i, c in enumerate(chars))
input_len=len(processed_inputs)
vocab_len=len(chars)
print ("Total number of characters:", input_len)
print ("Total vocab:", vocab_len)

seq_length = 100
x_data = []
y_data = []
for i in range(0, input_len - seq_length, 1):
    
    in_seq = processed_inputs[i:i + seq_length]

    
    out_seq = processed_inputs[i + seq_length]
    
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

n_patterns=len(x_data)
print(n_patterns)

X = numpy.reshape(x_data, (n_patterns, seq_length, 1))
X = X/float(vocab_len)
y = np_utils.to_categorical(y_data)


model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "model_weights_saved2.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
desired_callbacks = [checkpoint]
Batch=256
#model.fit(X, y, epochs=20, batch_size=Batch, callbacks=desired_callbacks)

filename = "model_weights_saved2.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

num_to_char = dict((i, c) for i, c in enumerate(chars))
start = numpy.random.randint(0, len(x_data) - 1)
pattern = x_data[start]
print("Random Seed:")
print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")

for i in range(100):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab_len)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = num_to_char[index]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]
