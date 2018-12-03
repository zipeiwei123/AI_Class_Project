###
# rnn.py
# This file was created by modifying an example provided by keras, 
# which can be found here: 
# https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
#
# Modified by: Samantha Kerkhoff, samantha_kerkhoff@student.uml.edu
###

from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
import random
import os
import sys
import xml.etree.ElementTree as ET
import preprocess


file_path = "D:\\data\\datascience.stackexchange.com\\Comments.xml"
save_path = "D:\\data\\saved\\stackexchangetest.h5"


# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Generates the next text after an input
def get_response(sentence_array, num_item):
    generated = ''
    
    for i in range(num_item):
        x_pred = np.zeros((1, maxlen, len(words)))
        for t, word in enumerate(sentence_array):
            if word in word_indices:
                x_pred[0, t, word_indices[word]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, 1.0)
        next_word = indices_word[next_index]

        generated += next_word
        sentence_array = sentence_array[1:] + [next_word]
    return generated

# Function invoked at end of each epoch. Prints generated text.
def on_epoch_end(epoch, _):
    model.save(save_path)
        
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text_array) - maxlen - 1)
    sentence_array = text_array[start_index: start_index + maxlen]
    
    generated = ''
    for w in sentence_array:
        generated += w
        
    print('Message:')
    print(generated)
    print('Response:')

    generated += get_response(sentence_array, 50)
            
    print(generated)
    print()

# Take in user input and interact with model        
def converse():
    while True:
        # Function invoked at end of each epoch. Prints generated text.
        user_input = input('>> ')
        sentence_array = preprocess.tokenize(user_input)

        print('Response:')
        print(get_response(sentence_array, 50))
        print()

# Get total vocab
tree = ET.parse(file_path)
root = tree.getroot()
full_text = ''
words = set([])
count = 0

for row in root:
    full_text += row.get('Text')
    count += 1
    
    if count % 1000 == 0:
        text_array = preprocess.tokenize(full_text)
        full_text = ''        
        words = words.union(set(text_array))

words = sorted(words)
print('Number of unique words:', len(words)) # 27923
        
word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

# Build model
maxlen = 40
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model = None

if os.path.isfile(save_path):
    print('Loading model from saved path.')
    model = load_model(save_path)
else:
    model = Sequential()    
    model.add(LSTM(128, input_shape=(maxlen, len(words))))
    model.add(Dense(len(words), activation='softmax'))
        
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Count through rows
tree = ET.parse(file_path)
root = tree.getroot()
full_text = ''
count = 0
num_rows = 400

for row in root:
    full_text += row.get('Text')
    count += 1
    
    if count % num_rows == 0:
        print('Training on rows', (count - num_rows), 'to', count)
        text_array = preprocess.tokenize(full_text)
        full_text = ''

        # cut the text in semi-redundant sequences of maxlen characters
        step = 3
        sentences = []
        next_words = []
        for i in range(0, len(text_array) - maxlen, step):
            sentences.append(text_array[i: i + maxlen])
            next_words.append(text_array[i + maxlen])
        print('nb sequences:', len(sentences))

        print('Vectorization...')
        x = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
        y = np.zeros((len(sentences), len(words)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence):
                if word in word_indices:
                    x[i, t, word_indices[word]] = 1
            if next_words[i] in word_indices:
                y[i, word_indices[next_words[i]]] = 1

        # Train model
        model.fit(x, y, batch_size=128, epochs=20, callbacks=[print_callback])
                
        del x
        del y
     
converse()