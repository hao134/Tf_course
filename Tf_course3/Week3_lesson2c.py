import numpy as np
import json
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocav_size = 1000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000

with open('./sarcasm.json', 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

training_sentences = sentences[:training_size]
training_labels = labels[:training_size]
testing_sentences = sentences[training_size:]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words = vocav_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

# training_sequences = tokenizer.texts_to_sequences(training_sentences)
# training_padded = pad_sequences(training_sequences, maxlen = max_length, padding = padding_type, truncating=trunc_type)
# testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
# testing_padded = pad_sequences(testing_sequences, maxlen = max_length, padding = padding_type, truncating=trunc_type)
#
# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocav_size, embedding_dim, input_length= max_length),
#     tf.keras.layers.Conv1D(128, 5, activation='relu'),
#     tf.keras.layers.GlobalMaxPooling1D(),
#     tf.keras.layers.Dense(24, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
# model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# model.summary()
#
# num_epochs = 50
#
# training_padded = np.array(training_padded)
# training_labels = np.array(training_labels)
# testing_padded = np.array(testing_padded)
# testing_labels = np.array(testing_labels)
#
# history = model.fit(training_padded, training_labels, epochs = num_epochs, validation_data=(testing_padded, testing_labels), verbose = 1)

model = tf.keras.models.load_model("/tmp/week3_lesson2c.h5")

sentence = ["granny starting to fear spiders in the garden might be real", "game of thrones season finale showing this sunday night","mom starting to fear son's web series closest thing she will have to grandchild","top snake handler leaves sinking huckabee campaign"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding = padding_type, truncating=trunc_type)
print(model.predict(padded))