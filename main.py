import tensorflow as tf
from tensorflow import keras

data = keras.datasets.mnist
(input_train, output_train), (input_test, output_test) = data.load_data()

model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(392, activation="relu"),
    tf.keras.layers.Dense(196, activation="relu"),
    tf.keras.layers.Dense(98, activation="relu"),
    tf.keras.layers.Dense(49, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(input_train, output_train, epochs=10)
model.evaluate(input_test, output_test)

