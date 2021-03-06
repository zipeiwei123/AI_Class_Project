The files in this directory are used to format a dataset, model a neural 
network, and train the neural network model on that dataset.  The files 
it contains are:

preprocess.py - This takes in a text string and converts it into an array of
words, whitespace, and punctuation.  This was written by Samantha Kerkhoff.

rnn.py - This file builds the recurrent neural network model.  It takes the
path of a dataset, pulls out the needed text, and formats it using
preprocess.py.  It then trains the model on this data, printing out examples
of the current state as it goes.  Finally, it gives the user an option to 
enter in text and examine the type of output that is generated.  This file was 
created by modifying an example provided by keras, which can be found here: 
https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

At the moment, we are using the Stack Exchange Data Dump provided by the 
Stack Exchange network at this link https://archive.org/details/stackexchange
for our dataset.  In particular, we are using the data from the data science
stack exchange, because we wanted to focus on AI and Machine Learning
conversational topics.