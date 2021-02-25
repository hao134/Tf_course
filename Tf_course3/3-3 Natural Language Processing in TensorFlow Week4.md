# 3-3 Natural Language Processing in TensorFlow Week4
###### tags: `tf_exam`

### 1. Deal with a whole article
https://reurl.cc/ZQrGAl
Part of the article:
![](https://i.imgur.com/X4So0JE.png)
```python=
tokenizer = Tokenizer()
data = "In the town of Athy one Jeremy Lanigan \n Battered away til he hadnt a ..."

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

print(tokenizer.word_index)
print(total_words)
```
![](https://i.imgur.com/topGyGs.png)
The next work, we tranform our article into 
this form
![](https://i.imgur.com/rVAJVvL.png)
```python=
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
xs, labels = input_sequences[:,:-1], input_sequences[:, -1]

# one-hot encoding
ys = tf.keras.utils.to_categorical(labels, num_classes = total_words)
```
Construct the model:
```python=
model = Sequential()
model.add(Embedding(total_words, 64, input_length = max_sequence_len - 1))
model.add(Biderectional(LSTM(20)))
model.add(Dense(total_words, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
history = model.fit(xs, ys, epoch = 500, verbose = 1)
```
predict:
```python=
seed_text = "Laurence went to dublin"
next_words = 100
  
for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
	predicted = model.predict_classes(token_list, verbose=0)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	seed_text += " " + output_word
print(seed_text)
```
result:
Laurence went to dublin as plenty as water water water rose mchugh mchugh mchugh rose mchugh mchugh wall hearty ... 

### 2. Deal with irish lytris text
https://reurl.cc/GdojbD
The data preprocessing is nothing different from the before notebook. In this notebook, the model is a bit different from the before model. we can see it:
```python=
model = Sequential()
model.add(Embedding(total_words, 100, input_length = max_sequence -1))
model.add(Biderectional(LSTM(150)))
model.add(Dense(total_words, activation = 'softmax'))
# split the optimizer part independently
adam = Adam(lr = 0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
# early stop 
# from tensorflow.keras.callbacks import EarlyStopping
# earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 5, verbose = 0, mode= 'auto')
model.summary()
```
The result:
![](https://i.imgur.com/twnCCKP.png)

### 3. Exercise:
Question:
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP_Week4_Exercise_Shakespeare_Question.ipynb
Answer:
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%203%20-%20NLP/NLP_Week4_Exercise_Shakespeare_Answer.ipynb#scrollTo=PRnDnCW-Z7qv
