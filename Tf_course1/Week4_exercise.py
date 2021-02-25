import tensorflow as tf
import os
import zipfile

DESIRED_ACCURACY = 0.999

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.999):
            print("\nReached accuracy 99.9%, so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),input_shape = (150, 150, 3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255.)
train_generator = train_datagen.flow_from_directory(
    directory= "./happy_sad_dataset/",
    target_size = (150, 150),
    class_mode = 'binary',
    batch_size = 10
)
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr = 1e-3),
              metrics=['accuracy'])
model.fit(
    train_generator,
    #steps_per_epoch=8,
    epochs = 30,
    callbacks = [callbacks]
)