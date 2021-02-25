# 3-1 Natural Language Processing in TensorFlow Week 1 and Week 2
###### tags: `tf_exam`
https://hackmd.io/imFhCbZaQO-_dY6FTWFQ0Q


### 1. Tokenizer -> word to index
num_words: the maximum number of words to keep, based on word frequency. Only the most common num_words-1 words will be kept. 
```python=
from tensorflow.keras.preprocessing.text import Tokenizer
sentences = [
    'i love my dog',
    'I, love my cat',
    'You love my dog!'
]
tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
```
![](https://i.imgur.com/53NKspK.png)

### 2. Word to index, Padding, ....
https://reurl.cc/6yakvy
Word to index, padding and the word not tokenize.
```python=
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprecessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]
# oov_token -> when the test text not tokenize, use the word assigned in oov_token instead
tokenizer = Tokenizer(num_words = 100, oov_token = "<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequens(sequences, maxlen = 5)
print("\nWord Index = ", word_index)
print("\nSequences =", sequences)
print("\nPadded Sequences:")
print(padded)

# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence =", test_seq)

padded = pad_sequences(test_seq, maxlen =10)
print("\n Padded Test Sequence")
print(padded)
```
![](https://i.imgur.com/8Axwi6X.png)

### 3. Week 1's Exercise:
Use a text to tokenize and practice all the technique used in first week.
https://reurl.cc/0Dxg7l

---
## Week 2
### Deal with imdb review text
https://reurl.cc/v5gD8e
```python=
# download data from tensorflow dataset
import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info = True, as_supervised = True)

import numpy as np
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s, l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))
    training_labels.append(l.numpy)
for s,l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())
  
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
```
Tokenize the text
```python=
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = "<OOV>"

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen = max_length, truncating = trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen = max_length)
```
decode word index -> word
```python=
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[3]))
print(training_sentences[3])
```
![](https://i.imgur.com/8VpKwJy.png)

construct the model (simple dnn model)
```python=
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
```
![](https://i.imgur.com/6CJio7Z.png)

After training
```python=
num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)=  (10000, 16)
```
![](https://i.imgur.com/NAnQyEV.png)
see the embedding layer's documents:
https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding

### 2. Another datasets -> sarcasm dataset
https://reurl.cc/L07aW9
Datasets preprocessing are nothing different to before notebook, 
The new thing in this notebook is adding the GlobalAveragePooling1D layer 
istead of the Flatten() layer
```python=
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
```

### 3. imdb subwords8k
https://reurl.cc/L07azy
8000 vocabulary 
subword -> tensorflow -> tensor / flow 
see the code:
```python=
import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True)

tokenizer = info.features['text'].encoder

sample_string = 'Tensorflow, from basics to mastery'

tokenized_string = tokenizer.encode(sample_string)
print('Tokenized string is {}'.format(tokenized_string))

original_string = tokenizer.decode(tokenized_string)
print('The original string is {}'.format(original_string))
```
![](https://i.imgur.com/5FToiSA.png)

```python=
for ts in tokenized_string:
    print('{} --> {}'.format(ts, tokenizer.decode))
```
![](https://i.imgur.com/cLIGCyK.png)

### 4. Exercise
Q:
https://reurl.cc/qm1GxD
A:
https://reurl.cc/NXZYv5
BBS NEWS TEXT -> 6 classes classification task
important:
1. deal with the data:
```python=
sentences = []
labels = []
with open("/tmp/bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " " # ex: stopword "all" -> token " all "
            sentence = sentence.replace(token, " ")
        sentences.append(sentence)
```
2. construct the model:
```python=
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length = max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation = 'relu'),
    tf.keras.layers.Dense(6, activaiton = 'softmax')
])
# use sparse_categorical_crossentropy, because we have 6 classes
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
```