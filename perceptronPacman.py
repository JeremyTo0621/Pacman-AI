# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import util
from pacman import GameState
import random
import numpy as np
from pacman import Directions
import math
import numpy as np
from featureExtractors import FEATURE_NAMES
import matplotlib.pyplot as plt


PRINT = True


class PerceptronPacman:

    def __init__(self, num_train_iterations=20, learning_rate=1):

        self.max_iterations = num_train_iterations
        self.learning_rate = learning_rate

        # A list of which features to include by name. To exclude a feature comment out the line with that feature name
        feature_names_to_use = [
            'closestFood',
            'closestFoodNow',
            'closestGhost',
            'closestGhostNow',
            'closestScaredGhost',
            'closestScaredGhostNow',
            'eatenByGhost',
            'eatsCapsule',
            'eatsFood',
            "foodCount",
            'foodWithinFiveSpaces',
            'foodWithinNineSpaces',
            'foodWithinThreeSpaces',
            'furthestFood',
            'numberAvailableActions',
            "ratioCapsuleDistance",
            "ratioFoodDistance",
            "ratioGhostDistance",
            "ratioScaredGhostDistance" 
        ]

        # we start our indexing from 1 because the bias term is at index 0 in the data set
        feature_name_to_idx = dict(zip(FEATURE_NAMES, np.arange(1, len(FEATURE_NAMES) + 1)))

        # a list of the indices for the features that should be used. We always include 0 for the bias term.
        self.features_to_use = [0] + [feature_name_to_idx[feature_name] for feature_name in feature_names_to_use]

        "*** YOUR CODE HERE ***"
        # Initialize weights for input to hidden and hidden to output layers
        self.weights_input_hidden = np.random.randn(len(self.features_to_use), 50) # hidden layer nodes
        self.weights_hidden_output = np.random.randn(50, 1) # Output layer

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def predict(self, feature_vector):
        """
        This function should take a feature vector as a numpy array and pass it through your perceptron and output activation function

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.
        """
        # filter the data to only include your chosen features. We might not need to do this if we're working with training data that has already been filtered.
        if len(feature_vector) > len(self.features_to_use):
            vector_to_classify = feature_vector[self.features_to_use]
        else:
            vector_to_classify = feature_vector

        "*** YOUR CODE HERE ***"
        hidden_layer_input = np.dot(vector_to_classify, self.weights_input_hidden)  # Input to hidden layer
        hidden_layer_output = self.activationHidden(hidden_layer_input)  # Activation for hidden layer
        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)  # Hidden to output layer
        return self.activationOutput(output_layer_input)  # Activation for output layer

    def activationHidden(self, x):
        """
        Implement your chosen activation function for any hidden layers here.
        """

        "*** YOUR CODE HERE ***"
        return np.maximum(0, x)  # ReLU activation function
        

    def activationOutput(self, x):
        """
        Implement your chosen activation function for the output here.
        """

        "*** YOUR CODE HERE ***"
        return 1 / (1 + np.exp(-x))  # Sigmoid activation function

    def evaluate(self, data, labels):
        """
        This function should take a data set and corresponding labels and compute the performance of the perceptron.
        You might for example use accuracy for classification, but you can implement whatever performance measure
        you think is suitable. You aren't evaluated what you choose here.
        This function is just used for you to assess the performance of your training.

        The data should be a 2D numpy array where each row is a feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, labels[1]
        is the label for the feature at data[1]
        """

        # filter the data to only include your chosen features
        X_eval = data[:, self.features_to_use]

        "*** YOUR CODE HERE ***"
        predictions = np.array([1 if self.predict(x) >= 0.5 else 0 for x in X_eval])
        accuracy = np.mean(predictions == labels)
        loss = np.mean((predictions - labels) ** 2)  # Mean Squared Error
        return loss, accuracy

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        This function should take training and validation data sets and train the perceptron

        The training and validation data sets should be 2D numpy arrays where each row is a different feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The training and validation labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, trainingLabels[1]
        is the label for the feature at trainingData[1]
        """

        # filter the data to only include your chosen features. Use the validation data however you like.
        X_train = trainingData[:, self.features_to_use]
        X_validate = validationData[:, self.features_to_use]

        "*** YOUR CODE HERE ***"
        for iteration in range(self.max_iterations):
            for i in range(len(X_train)):
                x = X_train[i]
                y = trainingLabels[i]
                
                # Forward pass
                hidden_layer_input = np.dot(x, self.weights_input_hidden)
                hidden_layer_output = self.activationHidden(hidden_layer_input)
                output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
                y_pred = self.activationOutput(output_layer_input)

                # Backward pass
                error = y - y_pred
                output_delta = error * (y_pred * (1 - y_pred))  # Derivative of sigmoid

                hidden_error = output_delta.dot(self.weights_hidden_output.T)
                hidden_delta = hidden_error * (hidden_layer_output > 0)  # Derivative of ReLU

                # Update weights
                self.weights_hidden_output += self.learning_rate * hidden_layer_output.reshape(-1, 1) * output_delta
                self.weights_input_hidden += self.learning_rate * x.reshape(-1, 1) * hidden_delta

            # Calculate and store training metrics
            train_loss, train_accuracy = self.evaluate(X_train, trainingLabels)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)

            # Calculate and store validation metrics
            val_loss, val_accuracy = self.evaluate(X_validate, validationLabels)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            if PRINT:
                print(f'Epoch {iteration + 1}/{self.max_iterations}, '
                    f'Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, '
                    f'Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
        self.plot_metrics()
    
    def plot_metrics(self):
        """Plot training and validation loss and accuracy."""
        epochs = range(1, self.max_iterations + 1)

        # Plotting loss
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss', color='blue')
        plt.plot(epochs, self.val_losses, label='Validation Loss', color='orange')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Plotting accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy', color='blue')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy', color='orange')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def save_weights(self, weights_path):
        """
        Saves your weights to a .model file. You're free to format this however you like.
        For example with a single layer perceptron you could just save a single line with all the weights.
        """
        "*** YOUR CODE HERE ***"
        with open(weights_path, "wb") as f:
            np.save(f, self.weights_input_hidden)
            np.save(f, self.weights_hidden_output)

    def load_weights(self, weights_path) -> None:
        """
        Loads your weights from a .model file.
        Whatever you do here should work with the formatting of your save_weights function.
        """
        "*** YOUR CODE HERE ***"
        with open(weights_path, "rb") as f:
            self.weights_input_hidden = np.load(f)
            self.weights_hidden_output = np.load(f)





