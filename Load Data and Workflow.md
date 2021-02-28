# Load Data and Workflow
https://hackmd.io/i387-QxbRYa9pSyXRh2btg
###### tags: `tf_exam`

1. Mnist data (fashion mnist or digit mnist):
shape (60000, 28, 28) -> (# of data, x_size, y_size)

(1). Use DNN model:
```python=
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

(2). Use Conv model:
remainder expand the dimension after the last axis
```python=
x_train = tf.expand_dims(x_train, axis = -1)
```

2. 沒有劃分train 和 validation 的影像資料：
請參考
https://github.com/hao134/Tf_course/blob/main/Tf_course2/Week2_exercise.py

3. 影像資料以csv擋儲存：

![](https://i.imgur.com/eVjP93T.png)

說明：第一個column 是數據的labels，共有26類，後面的是影像的資料總共784維，大小是 28 x 28

```python=
def get_data(filename):
    with open(filename) as training_file:
        reader = csv.reader(training_file, delimiter = ',')
        imgs = []
        labels = []
        
        # do not read the first row
        next(reader)
        for row in reader:
            label = row[0]
            data = row[1:]
            img = np.array(data).reshape((28, 28))
            
            imgs.append(img)
            labels.append(label)
        images = np.array(imgs).astype(float)
        labels = np.array(labels).astype(float)
# load data
training_images, training_labels = get_data('./archive/sign_mnist_train.csv')
testing_images, testing_labels = get_data('./archive/sign_mnist_test.csv')
# add the dimension
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)
```

4. Imdb_reviews
(1) train/validation
```python=
import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info = True, as_supervised = True)

train_data, validation_data = imdb['train'], imdb["test"]
training_sentences = []
training_labels = []

validation_sentences = []
validation_labels =[]

for s, l in train_data:
    training_sentences.append(s.numpy.decode('utf8'))
    training_labels.append(l.numpy())
for s, l in validation_data:
    validation_sentences.append(s.numpy.decode('utf8'))
    validation_labels.append(s.numpy())
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)
```
(2) train/validation/test
```python=
train_data, validation_data, test_data = tfds.load(
    name = 'imdb_reviews',
    split = ('train[:60%]', 'train[60%:]', 'test'),
    as_supervised = True)

training_sentences = []
training_labels = []

validation_sentences = []
validation_labels = []

testing_sentences = []
testing_labels = []

for s, l in train_data:
    training_sentences.append(s.numpy().decode('utf8'))
    # or training_sentences.append(str(s.numpy()))
    training_labels.append(l.numpy())
for s, l in validation_data:
    validation_sentences.append(s.numpy().decode('utf8'))
    validation_labels.append(l.numpy())
for s, l in test_data:
    testing_sentences.append(s.numpy().decode('utf8'))
    testing_labels.append(l.numpy())
    
training_labels_final = np.array(training_labels)
validation_labels_final = np.array(validation_labels)
testing_labels_final = np.array(testing_labels)
```

5. Real data Sarcasm data(text json):
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week2_lesson2.py
![](https://i.imgur.com/ZddyBAo.png)
```python=
with open("./sarcasm.json",'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]
```
6. Real data BBC_Text(text csv):
https://github.com/hao134/Tf_course/blob/main/Tf_course3/Week2_exercise.py
![](https://i.imgur.com/Y9Sui2u.png)
說明： 這是bbc新聞的分類任務，category 是類別，text是文章內容，機器判斷文章內容來做分類任務。
```python=
sentences = []
labels = []
# 把stopwords禁掉，大概是一些介系詞，冠詞之類不影響分類任務的字
stopwords = ["a","about","above",...]
with open("./bbc-text.csv",'r') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    #不要讀第一行
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token," ")
        sentences.append(sentence)
        
train_sentences = sentences[:train_size]
train_labels = labels[:train_size]
validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]
```

7. 真實數據，二元分類(text csv):
https://nbviewer.jupyter.org/github/hao134/Tf_course/blob/main/Tf_course3/NLP_Course_Week_3_Exercise_Answer.ipynb
![](https://i.imgur.com/25SqknB.png)
說明：第五個column是text內容，第一個column是label，總共有兩類，一開始是'0'後來是'4'
```python=
corpus = []
with open('./training_cleaned.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter = ','):
    for row in reader:
        list_item = []
        list_item.append(row[5])
        this_label = row[0]
        if this_label == '0':
            list_item.append(0)
        else:
            list_item.append(1)
        corpus.append(list_item)
sentences = []
labels = []
import random
random.shuffle(corpus)
for x in range(training_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])
```

8. 詩詞處理，以莎士比雅為例子：
```python=
tokenizer = Tokenizer()
data = open('./sonnets.txt').read()

corpus = data.lower().split('\n')

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# pad_sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding = 'pre'))

# create predictors and label
predictors, label = input_sequences[:,:-1], input_sequences[:,-1]
label = tf.keras.utils.to_categorical(label, num_classes = total_words)
```

9. Tokenize, padding ...，處理texts 的流程：
(1）變數：
```python=
embedding_dim = 100 # embedding dim
max_length = 120 # padding 最大長度
trunc_type = 'post' # 過長要截斷前面還後面
padding_type = 'post' # 補0要補前還補後
vocab_size = 1000 # tokenize 最常出現的前1000個字，不指定則會tokenize全部的字
oov_tok = '<OOV>' # 沒有背tokenize的字用這個表示
```
(2)load data:
前面提過了
(3) Tokenize and padding
```python=
tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok)
# 套用設定的方法來tokenize 資料
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, padding = padding_type, maxlen = max_length)
```
(4)建模(Embedding)：
model =tf.keras.Sequential([
    tf.keras.Embedding(vocab_size, embedding_dim, input_length = max_length),
    ...
])

10. 真實時序資料：Sunspots
![](https://i.imgur.com/fN94eth.png)
說明，第一個column是順序，當作label，第三個column是值，當作資料點：
```python=
import csv
time_step = []
sunspots = []

with open("./sunspots.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter = ",")
    # don't read the first line
    next(reader)
    for row in reader:
        sunspots.append(row[2])
        time_step.append(row[0])
series = np.array(sunspots)
time = np.array(time_step)
```

11. Example From tensorflow website:
他們提供了一套用他們套件預處理的方法，但如果自己會的話，就用自己的方法無所謂：
* Image classification:
https://www.tensorflow.org/tutorials/images/classification
* Word Imbedding
https://www.tensorflow.org/tutorials/text/word_embeddings
