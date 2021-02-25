import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("accuracy")>0.95):
            print("\nReached 95 % acccuracy, so cancelling training")
            self.model.stop_training = True
callbacks = myCallback()
x_train = tf.expand_dims(x_train, axis = -1)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),input_shape=(28, 28, 1), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    #tf.keras.layers.Dense(10, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 30, callbacks=[callbacks])
x_test = tf.expand_dims(x_test, axis = -1)
model.evaluate(x_test, y_test)

