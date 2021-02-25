# 4-2 Sequences, Time Series and Prediction Week3 and Week4
###### tags: `tf_exam`
https://hackmd.io/7iA5FsWLRZSr9pGka6ukrQ

### 1.Same Sequeutial data called "series" using RNN model
https://reurl.cc/a51dG3
The RNN model
![](https://i.imgur.com/hLvtQWc.png)
notice the input data, the batch size is equal to window size,
it can be assigned to any value. Thus, we can design our model as follow:
```python=
# clear the backend variables
tf.keras.backend.clear_session()
# fixed the seed to let the result fixed
tf.random.set_seed(51)
np.random.seed(51)

train_set = window_dataset(x_train, window_size = 20, batch_size = 128, shuffle_buffer = shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -1),
                            input_shape = [None]),
    tf.keras.layers.SimpleRNN(40, return_sequence = True),
    # equal to keras.layers.SimpleRNN(40, return_sequence = True,
    #                                 input_shape = [None, 1])
    # None because we accept any Batch size, here we set batch_size equal to 20 (window_size)
    tf.keras.layers.SimpleRNN(40),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 100.0)
])

lr_schdule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr = 1e-8, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(),
              optimizer = optimizer,
              metrics = ['mae'])
history = model.fit(train_set, epochs = 100, callbacks = [lr_schedule])

plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])
```
![](https://i.imgur.com/cbMbiiJ.png)
We can pich up the learning rate where the loss is lower. 
For example, lr = 5e-5. 
re-train the model using this learning rate to make the results better.

### 2. Do the same thing using LSTM
https://reurl.cc/raLvZx
```python=
model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])
```
The result:
![](https://i.imgur.com/nkW011k.png)

### 3. Exercise 
* Question:
https://reurl.cc/v5k4rL 
* Answer:
https://reurl.cc/0DXxd9
---
## Week 4

### 1.Before LSTM layers, using Convolution layer
https://reurl.cc/e9OmpQ
The Squential dataset are the same. The Tensorflow deal with the sequential data's little code is a little different. The model how to predict sequential data's code is also put there.
```python=
def window_dataset(series, window_size, batch_size, shuffle_buffer):
    # add this code 
    series = tf.expand_dims(series, axis = -1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift = 1, drop_remainder = True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
    
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift = 1, drop_remainder = True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast 
```
Construct the model
```python=
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

window_size = 30
train_set = windowed_dataset(x_train, window_size, batch_size = 128, shuffle_buffer = shuffer_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters = 32, kernel_size= 5,
                        strides = 1, padding = 'causal',
                        activation = 'relu',
                        input_shape = [None, 1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences = True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences = True)),
    tf.keras.Dense(1),
    tf.keras.layers.Lambda(lambda x: x* 200)
])
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr = 1e-8, momentum = 0.9)
model.compile(loss = tf.keras.losses.Huber(),
              optimizer = optimizer,
              metrics = ['mae'])
history = model.fit(train_set, epochs = 100, callbacks = [lr_schedule])
```
Plot the results:
```python=
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 30])
```
![](https://i.imgur.com/q0tGbdt.png)
we can see the best lr = 1e-5
After epochs 500,
plot the prediced results
```python=
rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
```
![](https://i.imgur.com/BYCm7Yq.png)

### 2. Using DNN Train a Real-World data (Sunspots)
https://reurl.cc/MZ87XL

1. Plot the data:

![](https://i.imgur.com/fN94eth.png)

```python=
def plot_series(time, series, format = "-", start = 0, end = None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
import csv
time_step = []
sunspots = []

with open("/tmp/sunspots.csv") as csvfile:
    reader = csv.reader(csvfile, delimiter = ",")
    # don't read first line
    next(reader)
    for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))
series = np.array(sunspots)
time = np.array(time_step)
plt.figure(figsize = (10, 6))
plot_series(time, series)
```
![](https://i.imgur.com/XcY4NSP.png)

Total 3500 data potints, thus we split 3000 data points as train data
```python=
split_time = 3000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

window_size = 60
batch_size = 32
shuffle_buffer_size = 1000
```

define windowed_dataset and construct the model:

```python=
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift = 1, drop_remainder = True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
    
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size):

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, input_shape = [window_size], activation = "relu"),
    tf.keras.layers.Dense(10, activation = 'relu')
    tf.keras.layers.Dense(1)
])

model.compile(loss = "mse", optimizer = tf.keras.optimizers.SGD(lr = 1e-7, momentum = 0.9))
model.fit(dataset, epochs = 100, verbose = 0)
```
predicted result and predicted loss

```python=
forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
    
forecast = forecast[split_time - window_size:]
results = np.array(forecast)[:, 0, 0]

plt.figure(figsize = (10, 6))

plot_series(time_valid, x_valid)
plot_series(time_valid, results)
print(tf.keras.metrics.mean_absolute_error(x_valid, results).numpy())
```
![](https://i.imgur.com/ogllk0z.png)

### 3. Using LSTM Do the Same Task
https://reurl.cc/KxQvQy

new windowed dataset (add expand_dims), and model_forecast code
```python=
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis = -1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift = 1, drop_remainder = True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

# need special input setting, because we using CNN layer first
def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift = 1, drop_remainder = True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast
```
Construct model
```python=
window_size = 64
batch_size = 256
train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters = 32, kernel_size = 5,
                        strides = 1, padding = "causal",
                        activation = "relu",
                        input_shape = [None, 1]),
    tf.keras.layers.LSTM(64, return_sequences = True),
    tf.keras.layers.LSTM(64, return_sequences = True),
    tf.keras.layers.Dense(30, activation = 'relu'),
    tf.keras.layers.Dense(10, activarion = 'relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x * 400)
])
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr=1e-8, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
```
After training:
```python=
plt.semilogx(history.history["lr"], history.history["loss"])
plt.axis([1e-8, 1e-4, 0, 60])
```
![](https://i.imgur.com/Bz6AfaV.png)
change the lr = 1e-5
and retrain it:
```python=
rnn_forecast = model_forecast(model, series[...,np.newaxis], window_size)
rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]

plt.figure(figsize = (10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, rnn_forecast)
print(tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy())
```
![](https://i.imgur.com/LkBBNBH.png)

How to improve the training results,
1. change the batch_size
2. change the CNN model's filters
3. learning rate 
4. change the model(ex: add more LSTM layer, change the units in layers)

### 4. Week4's Exercise
* Question:
https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Exercise%20Question.ipynb
* Answer:
https://github.com/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Exercise%20Answer.ipynb