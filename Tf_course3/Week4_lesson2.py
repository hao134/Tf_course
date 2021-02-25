import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

tokenizer = Tokenizer()

data = open('./irish-lyrics-eof.txt').read()

corpus = data.lower().split('\n')

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)

input_sequences = []

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# pad sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_sequence_len, padding = 'pre'))

# create predictors and label
xs, labels = input_sequences[:,:-1], input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len-1))
# model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))
# model.add(tf.keras.layers.Dense(total_words, activation = 'softmax'))
# adam = tf.keras.optimizers.Adam(lr = 0.01)
# model.compile(loss= 'categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
# earlystop = tf.keras.callbacks.EarlyStopping(monitor = 'loss', min_delta=0, patience=5, verbose = 0, mode = 'auto')
# history = model.fit(xs, ys, epochs = 100, verbose = 1, callbacks=[earlystop])
# print(model.summary())
#
# import matplotlib.pyplot as plt
#
#
# def plot_graphs(history, string):
#   plt.plot(history.history[string])
#   plt.xlabel("Epochs")
#   plt.ylabel(string)
#   plt.show()
#
# plot_graphs(history, 'accuracy')

seed_text = "I've got a bad feeling about this"
next_words = 100

model = tf.keras.models.load_model('/tmp/week4_lesson2.h5')
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding = 'pre')
    predicted = model.predict_classes(token_list,verbose = 0)
    # predicted = np.argmax(model.predict(token_list), axis = -1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)