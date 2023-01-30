import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# importing mnist dataset
mnist = tf.keras.datasets.mnist
# Loading in testing and training dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalising the testing and training dataset to be within 1-0
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


# Creates an empty model that will allow you to add layers
model = tf.keras.models.Sequential()
# Flattens the input of the model to be 1-dimensional array
model.add(tf.keras.layers.Flatten())
# Creates a linked up layer with 128 neurons
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# The activation function is non-linear which allows the model to learn more complex functions
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# The output layer, since there are 10 different categories (numbers) there is 10 neurons in this layer
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

model.save('beginnerMNISTModel.model')
new_model = tf.keras.models.load_model('beginnerMNISTModel.model')

val_loss, val_acc = new_model.evaluate(x_test, y_test)
print(val_loss, val_acc)

predictions = new_model.predict([x_test])
print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()
