import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def plot_series(time, series, format = "-", start = 0, end = None, label = None):
    plt.plot(time[start:end], series[start:end], format, label = label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize = 14)
    plt.grid(True)

# Trend and Seasonality
def trend(time, slope = 0):
    return slope * time

time = np.arange(4 * 365 + 1)
baseline = 10
series = trend(time, 0.1)

plt.figure(figsize = (10, 6))
plot_series(time, series)
plt.show()

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude = 1, phase = 0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

baseline = 10
amplitude = 40
series = seasonality(time, period=365, amplitude = amplitude)
plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

#===================
slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period = 365, amplitude=amplitude)

plt.figure(figsize=(10,6))
plot_series(time, series)
plt.show()

# Noise
def white_noise(time, noise_level = 1, seed = None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

noise_level = 5
noise = white_noise(time, noise_level, seed = 42)
plt.figure(figsize=(10, 6))
plot_series(time, noise)
plt.show()

# Series + Noise
series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.show()

# spilt the data into training and validation data and see more example
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

def autocorrelation(time, amplitude, seed = None):
    rnd = np.random.RandomState(seed)
    phi1 = 0.5
    phi2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += phi1 * ar[step - 50]
        ar[step] += phi2 * ar[step - 33]
    return ar[50:] * amplitude

def autocorrelation(time, amplitude, seed = None):
    rnd = np.random.RandomState(seed)
    phi = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += phi * ar[step - 1]
    return ar[1:] * amplitude

series = autocorrelation(time, 10, seed=42)
plot_series(time[:200], series[:200])
plt.show()