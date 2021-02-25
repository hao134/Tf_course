# 2-2 Convolutional Neural Networks in TensorFlow week3 & week4
###### tags: `tf_exam`
https://hackmd.io/iB8W9uaQRU20x1841lQ6Mg

### 1. Transfer learning basic
https://reurl.cc/9ZZRzx
Try to train a transfer learning model on dog-v-cat dataset. The model we used is Inception model.

1. load InceptionV3 model
```python=
from tensorflow.keras.applications.inception_v3 import InceptionV3

pre_trained_model = InceptionV3(input_shape = (150, 150, 3),
                                include_top = False,
                                weights = None)
                                
# pre_download the model's weight
pre_trained_model.load_weighs(local_weights_file)

# frozen the model's weight, we use the model weights trained.
for layer in pre_trained_model.layers:
    layer.trainable = False

# adjust the last layer
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape', last_layer.output_shape)
last_output = last_layer.output
```
![](https://i.imgur.com/gg2ud1g.png)

2. change the last few last layers as our las layers
The pre_trained model's output need to change to our training task's need. so we change the last few layer.
```python=
from tensorflow.keras.optimizers import RMSprop
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1024 hidden units and ReLU activation
x = layers.Dense(1024, activation = 'relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation = 'sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(opimizer = RMSprop(lr = 1e-4),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
```

### 2. Week 3's exercise
https://reurl.cc/dVGzeg

### 3. Rock-Paper-Scissors dataset
https://reurl.cc/Q79Y6q
note: Multi-classes classification dataset
Data preparing are same as before procedure.
See the important part of this notebook:
```python=
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "/tmp/rps/"
training_datagen = ImageDataGenerator(
    rescale = 1/255,
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range =0.2,
    shear_range =0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest')
VALIDATION_DIR = "/tmp/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1/255)
train_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size = (150, 150),
    class_mode = 'categorical',
    batch_size = 126)
validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126)
```
Build model:
**Change the output layer from sigmoid to softmax**
**choose loss 'categorical_crossentropy' instead of binary crossentropy**
**for multi-classification task**
```python=
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    # second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # 3 rd convolution
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # 4 th convolution
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'),
    tf.kears.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5)
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(3, activation = 'softmax')
])

model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

history = model.fit(train_generator,
                    epochs = 25,
                    steps_per_epoch = 20,
                    validation_data = validation_generator,
                    verbose = 1,
                    validation_steps = 3)
model.save("rps.h5")

```

result:
![](https://i.imgur.com/1xpIfZz.png)

### 4. Week4's Exercise
https://reurl.cc/OX0QK7


important:
deal with the structural dataset -> csv data
![](https://i.imgur.com/eVjP93T.png)
```python=
def get_data(filename):
    with open(filename) as training_file:
        reader = csv.reader(training_file, felimiter = ',')
        imgs = []
        labels = []
        
        # no read the first row
        next(reader)
        for row in reader:
            label = row[0]
            data = row[1:]
            img = np.array(data).reshape((28, 28))
            
            imgs.append(img)
            labels.append(label)
        images = np.array(imgs).astype(float)
        labels = np.array(labels).astype(float)
        
    return images, labels

training_images, training_labels = get_data('sign_mnist_train.csv')
testing_images, testing_labels = get_data('sign_mnist_test.csv')

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)
```
![](https://i.imgur.com/7r9x20T.png)

2. add dimension to the data to fit covolution input:
```python=
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    rescale=1. / 255)

print(training_images.shape)
print(testing_images.shape)
```
![](https://i.imgur.com/Xi7pqIf.png)

3. construct the model and train it:
```python=
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(26, activation=tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_datagen.flow(training_images, training_labels, batch_size=32),
                              steps_per_epoch=len(training_images) / 32,
                              epochs=15,
                              validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32),
                              validation_steps=len(testing_images) / 32)

model.evaluate(testing_images, testing_labels)
```
