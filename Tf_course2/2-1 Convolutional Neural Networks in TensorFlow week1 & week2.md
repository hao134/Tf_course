# 2-1 Convolutional Neural Networks in TensorFlow week1 & week2
###### tags: `tf_exam`
https://hackmd.io/JQDepau-TSakWfjCOrYJLA

### 1. Dog-vs cat dataset
There has no new from this notebook, just see it.
https://reurl.cc/o9dYL3

### 2. Week1's Exercise
Good example for practing creating the training and validation set.
* question:
https://reurl.cc/nn0YlD
* answer:
https://reurl.cc/pmyYK8

1. import library
```python=
import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image ImageDataGenerator
from shutil import copyfile
```
2. Download dataset
```python=
!wget --no-check-certificate \
    "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
    -O "/tmp/cats-and-dogs.zip"

local_zip = '/tmp/cats-and-dogs.zip'
zip_ref   = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
```
3. Split and Create folder for training data and validation data
```python=
try:
    os.mkdir('/tmp/cats-v-dogs')
    os.mkdir('/tmp/cats-v-dogs/training')
    os.mkdir('/tmp/cats-v-dogs/testing')
    os.mkdir('/tmp/cats-v-dogs/training/cats')
    os.mkdir('/tmp/cats-v-dogs/training/dogs')
    os.mkdir('/tmp/cats-v-dogs/testing/cats')
    os.mkdir('/tmp/cats-v-dogs/testing/dogs')
except OSError:
    pass
```

**4. Define how to split data**
```python=
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            file.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")
    
    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[-testing_length:]
    
    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)
        
    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)

CAT_SOURCE_DIR = "/tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "/tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "/tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "/tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "/tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "/tmp/cats-v-dogs/testing/dogs/"

split_size = 0.9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
```
5. Construct the model
```python=
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.compile(optimizer = RMSprop(lr = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])
```
**6. DATA GENERATOR**
```python=
TRAINING_DIR = "/tmp/cats-v-dogs/training/"
train_datagen = ImageDataGenerator(rescale = 1.0/255.0)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size = 100,
                                                    class_mode = 'binary',
                                                    target_size = (150, 150))
VALIDATION_DIR = "/tmp/cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale = 1.0 / 255.0)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                             batch_sie = 100,
                                                             class_mode = 'binary',
                                                             target_size = (150, 150))

```

7. Training
```python=
history = model.fit(train_generator,
                    epochs = 50,
                    verbose = 1,
                    validation_data = validation_generator)
```

### 3. Augmentation
https://reurl.cc/jqq1zM
Augmentation help the training process not to overfitting. Let us use dog-v-cat dataset to see how augmentation on data's impact.
1. no augmentation
```python=
ImageDataGenerator(rescale = 1.0/ 255.0)
```
heavy overfitting
![](https://i.imgur.com/lDfQKyW.png)

2. basic augmentation
```python=
train_datagen = ImageDataGenerator(
    rescale = (1./255)
    rotation_range = 40, # give degree range (0-180) to random rotate picture
    width_shift_range = 0.2, # give fraction to randomly translate pictures vertically or horizontally
    height_shift_range = 0.2,
    shear_range = 0.2, # randomly applying shearing transformations
    zoom_range = 0.2, # randomly zooming inside pictures
    horizontal_flip = True, # randomly flipping half of the images horizontally.
    fill_mode = 'nearest' # strategy used for filling in newly created pixels, which can appear after a rotation shift.
    )
# no need augmentation for test data
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(150, 150),  # All images will be resized to 150x150
        batch_size=20,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

history = model.fit(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=100,
      validation_data=validation_generator,
      validation_steps=50,  # 1000 images = batch_size * steps
      verbose=2)
```
![](https://i.imgur.com/0lM1R9T.png)


### 4. Use same augumentation on horse-v-human dataset
https://reurl.cc/pmmZz4

### 5. Exercise -> week one's dataset add augumentation
https://reurl.cc/Q77Xqp