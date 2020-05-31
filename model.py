from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

class model:

    def plot_curve(self,epochs, hist, list_of_metrics):
        """Plot a curve of one or more classification metrics vs. epoch."""
        # list_of_metrics should be one of the names shown in:
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Value")

        for m in list_of_metrics:
            x = hist[m]
            plt.plot(epochs[1:], x[1:], label=m)

        plt.legend()

        print("Loaded the plot_curve function.")

    def create_model(self,my_learning_rate):
        """Create and compile a deep neural net."""

        # All models in this course are sequential.
        model = tf.keras.models.Sequential()

        # The features are stored in a two-dimensional 28X28 array.
        # Flatten that two-dimensional array into a a one-dimensional
        # 784-element array.
        model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

        # Define the first hidden layer.
        model.add(tf.keras.layers.Dense(units=256, activation='relu'))

        model.add(tf.keras.layers.Dense(units=128, activation='relu'))

        # Define a dropout regularization layer.
        model.add(tf.keras.layers.Dropout(rate=0.2))

        # Define the output layer. The units parameter is set to 10 because
        # the model must choose among 10 possible output values (representing
        # the digits from 0 to 9, inclusive).
        #
        # Don't change this layer.
        model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

        # Construct the layers into a model that TensorFlow can execute.
        # Notice that the loss function for multi-class classification
        # is different than the loss function for binary classification.
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                      loss="sparse_categorical_crossentropy",
                      metrics=['accuracy'])

        return  model

    def train_model(self,model, train_features, train_label, epochs,
                    batch_size=None, validation_split=0.1):
        """Train the model by feeding it data."""

        history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                            epochs=epochs, shuffle=True,
                            validation_split=validation_split)

        # To track the progression of training, gather a snapshot
        # of the model's metrics at each epoch.
        epochs = history.epoch
        hist = pd.DataFrame(history.history)

        return epochs, hist

