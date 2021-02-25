# 4-1 Sequences, Time Series and Prediction Week1 & Week2
###### tags: `tf_exam`
https://hackmd.io/9LXqJ4z2Sx28Ijg1io40qg

### 1. The notebook go through a few scenarios for time series. This notebook contains the code for that with a few little extras!
https://reurl.cc/9ZOL2Y

### 2. Before using Machine Learning, using stantard statistic technique to predict sequential data:
https://reurl.cc/g80yEQ

The next code block will set up the time series with seasonality, trend and a bit of noise.
```python=
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras

def plot_series(time, series, format = "-", start = 0, end = None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    
def trend(time, slope = 0):
    return slope * time
    
def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))
                    
def seasonality(time, period, amplitude = 1, phase = 0):
    """Reapeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def noise(time, noise_level = 1, seed = None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level
    
time = np.arange(4 * 365 + 1, dtype = "float32")
baseline = 10
series = trend(time, 0.1)
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = baseline + trend(time, slope) + seasonality(time, period = 365, amplitude = amplitude)
# Update with noise
series += noise(time, noise_level, seed = 42

plt.figure(figsize = (10, 6))
plot_series(time, series)
plt.show()
```
![](https://i.imgur.com/16gqwgz.png)

split the train and validation dataset
```python=
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]
plt.figure(figsize = (10, 6))
plot_series(time_train, x_train)
plt.show()

plt.figure(figsize = (10, 6))
plot_series(time_valid, x_valid)
plt.show()
```
![](https://i.imgur.com/XcmeoN2.png)
The next part is Naive forecast. It is very easy, see that in the notebook.

**Moving Average**
Forecasts the mean of the last few values.
It window_size = 1, then this is equivalent to naive forecast
```python=
def moving_average_forecast(series, window_size):
    forecast = []
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())
    return np.array(forecast)
    
# predict and plot it
moving_avg = moving_average_forecast(series, 30)[split_time - 30:]

plt_figure(figsize = (10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, moving_avg)
```
**Diff series**
That's worse than naive forecast! The moving average does not anticipate trend or seasonality, so let's try to remove them by using differencing. Since the seasonality period is 365 days, we will subtract the value at time t – 365 from the value at time t.
```python=
diff_series = (series[365:] - series[:-365])
diff_time = time[365:]
plt.figure(figsize = (10, 6))
plot_series(diff_time, diff_series)
plt.show()
```
![](https://i.imgur.com/0I16PAx.png)
Great, the trend and seasonality seem to be gone, so now we can use the moving average:
```python=
diff_moving_avg = moving_average_forecast(diff_size, 50)[split_sime - 365 - 50:]

plt.figure(figsize = (10, 6))
plot_series(time_valid, diff_series[split_time - 365:])
plot_series(time_valid, diff_moving_avg)
plt.show()
```
![](https://i.imgur.com/rajcSUf.png)
let's bring back the trend and seasonality by adding the past values from t – 365:
```python=
diff_moving_avg_plus_past = series[split_time - 365:-365] + diff_moving_avg

plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, diff_moving_avg_plus_past)
plt.show()
```
![](https://i.imgur.com/6WtUZkK.png)


### 3. Week1's Exercise:
Question:
https://reurl.cc/V3jxLR
Answer:
https://reurl.cc/NXpM0x

---
## Week2
### 1. Using Tensorflow to prepare Seaquential data's Features and labels
https://reurl.cc/3NoKR9
```python=
# prepare the data 0 ~ 9
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift = 1)
for window_dataset in dataset:
    for val in window_dataset:
        print(val.numpy(), end = " ")
    print()
```
![](https://i.imgur.com/TweTJTu.png)
```python=
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift = 1, drop_remainder = True)
dataset = dataset.flap_map(lambda window: window.batch(5))
# split out the label
dataset = dataset.map(lambda window: (window[:-1], iwndow[-1:]))
for x, y in dataset:
    print(x.numpy(), y.numpy())
```
![](https://i.imgur.com/VkFT08S.png)
create more dataset and labels
```python=
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift = 1, drop_remainder = True)
dataset = dataset.flap_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size = 10)
dataset = dataset.batch(2).prefeth(1)
for x, y in dataset:
    print("x =", x.numpy())
    print("y =", y.numpy())
```
![](https://i.imgur.com/kD9t5Xt.png)

### 2.Predict Sequential data with single neural network
https://reurl.cc/MZ03e3
1. Create a sequential data called "series". How to prepare it? See above, the code are the same code.
2. Use tensorflow to let "series" split into dataset and labels.
Just like this:
![](https://i.imgur.com/SpeGjEC.png)
```python=
window_size = 20
batch_size = 32
shuffle_buffer_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    # 20 dataset and 1 label
    dataset = dataset.window(window_size + 1, shift = 1, drop_remainder = True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer).map(lambda: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset
```
Construct the one layer simple DNN and train it
```python=
dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(dataset)
l0 = tf.keras.layers.Dense(1, input_shape = [window_size])
model = tf.keras.models.Sequential([l0])

model.compile(loss = "mse", optimizer = tf.keras.optimizers.SGD(lr = 1e-6, momentum = 0.9))
model.fit(dataset, epochs = 100, verbose = 0)

print("Layer weights {}".format(l0.get_weights())
```
![](https://i.imgur.com/4Exvrs7.png)
the predicted y eaual to l0's weight apply on data points x.

plot it
```python=
forecast = []
for time in range(len(series) - window_size):
    forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
forecast = forecast[split_time - window_size]
results = np.array(forecast)[:, 0, 0]

plt.figure(figsize = (10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, results)
```
![](https://i.imgur.com/6CNcHn7.png)

### 3.Predict Sequential data with Multi-layers neural network
https://reurl.cc/L0mQ2y
change the single layer to multi-layers, other are remain the same.
The important part needed to remainber is the reducing of the learning rate.
```python=
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape = [window_size], activation = "relu"),
    tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dnese(1)
])

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))
optimizer = tf.keras.optimizers.SGD(lr = 1e-8, momentum = 0.9)
model.compile(loss = "mse", optimizer = optimizer)
history = model.fit(dataset, epochs = 100, callbacks = [lr_schedule], verbose = 0)

lrs = 1e-8 * *(10 ** (np.arange(100) / 20))
plt.semilogx(lrs, history.history["loss"])
plt.axis([1e-8, 1e-3, 0, 300])
```
![](https://i.imgur.com/oeqnWNp.png)
we can find where the loss is the lowest, and set our learning rate at that place.

### 4. Week2's Exercise
Question:
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP_Week_2_Exercise_Question.ipynb
Answer:
https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP_Week_2_Exercise_Answer.ipynb#scrollTo=w552zJLoZ_jC



