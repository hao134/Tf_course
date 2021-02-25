import tensorflow as tf

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.998):
            print("\nReached 99.8% accuracy so cancelling training")
            self.model.stop_training = True
callbacks = MyCallback()

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0
training_images = tf.expand_dims(training_images, axis = -1)
test_images = tf.expand_dims(test_images, axis = -1)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])
model.evaluate(test_images, test_labels)