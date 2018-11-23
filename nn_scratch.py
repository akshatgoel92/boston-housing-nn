#--------------------------------#
# Author: Akshat Goel
# Date: 28-Oct-2018
# Description: Basic Neural Net
#--------------------------------#

# Import libraries 
import random 
import os 
import numpy as np

# Artificial Neural Net
# This Python class implements an artificial neural net
# Class variables are: 
# arch: Vector specifiying the number of neurons in each hidden layer
# layers: Length of the arch vector which tells us the no. of hidden layers
# weights: Impact of each input on neuron - initialized using a standard normal distribution
# learning_rate: Hyperparameter that controls size of gradient decent jumps at each iteration

class NeuralNet(object):

    # Divide into train and test data
    # Divide into features and labels
    # Initialize weights
    # Intialize no. of layers
    # Define architecture 
    def __init__(self, train, test, arch, rate, iterations): 

        # Separate training labels
        # Separate training features
        self.train_labels = train[-1]
        self.train_features = train[:-1]

        # Separate test labels
        # Separate test features
        self.test_labels = test[-1]
        self.test_features = train[:-1]
        
        # Store training data dimensions
        # n_x: No. of observations
        # n_y: No. of observations 
        # n_m: No. of features
        self.n_x = nrow(training)
        self.n_y = nrow(training)
        self.n_m = ncol(training)
        
        # Need to insert dimension of input layer
        # User inputs only architecture of hidden layers
        # Calculate the no. of layers only after this is done
        # This is so that we include the input layer in the total layers 
        self.arch = arch.insert(0, n_x)
        self.layers = range(1, len(arch) + 1)
        
        # Store the learning rate
        # Store the number of iterations
        self.rate = rate
        self.iterations = iterations
    
    # Input: Vector with no. of neurons in each layer
    # Output: Vector of initialized weights and biases
    def get_inital_weights(self):
        
        # This contains data for:
        # Architecture
        # Layers 
        arch = self.arch
        layers = self.layers

        W = {l: np.random.randn(arch[l], arch[l - 1]) for l in layers}
        b = {l: np.random.randn(arch[l-1], 1) for l in layers} 
        
        return(W, b)

    # Forward propagation:
    # Calculate the activations for each neuron
    # The activation function is given later below 
    # Each neuron either fires or doesn't
    def forward_propogation(self, X, Y, W, B):

        # Store the layers 
        # iterable
        layers = self.layers 
        L = len(layers)

        # This is the initial layer
        # where to store information 
        A = {0: X}
        Z = {1: X}
        
        # Now iterate over layers
        for l in layers:
            
            a = A[l-1]
            w = W[l)]
            b = B[l]
            
            Z[l] = np.sum(w @ a, b)
            A[l] = self.sigmoid(Z[l]))

        # Store the 'error' term from the last layer
        dZL = np.dot(A[L] - Y, self.sigmoid_prime(Z[L])) 

        return(dZL, A, W, b) 

    # Backward propogation
    # Propagate the output layer's error backward
    # Calculations are given by back-propagation equations
    # This allows us to see how cost is impacted by adjusting weights 
    # in previous layers 
    def backward_propogation(self, dZL, dAL):

        # Initalize dictionaries to 
        # store the derivatives
        L = len(self.arch)
        
        dA = {L: dAL}
        dZ = {L: dZL}
        dW = {}
        db = {}
        
        for layer in reversed(range(2, L)):

            dZ[l-1] = np.dot(W[l].T @ dZ[l], self.sigmoid_prime(Z[l-1])) 
            dW[l-1] = dZ[l] @ A[l].T
            db[l-1] = db[l]

        return(dW, db)
         
    # Cost function
    # Quadratic cost (Mean Squared Error)
    # Override this in other implementations
    # Input: Predictions vector, labels
    # Output: Cost 
    def get_cost(self, labels, preds):
    
        return(np.sum(np.power(labels - preds,2)))

    # Activation function
    # Sigmoid function 
    # Over-ride this in other implementations
    def sigmoid(self, z): 

        return(1.0/(1.0 + np.exp(-z)))

    # Cost derivative function
    # Sigmoid function derivative
    # Override this in other implementations
    # Input: -
    # Output: Activation derivative
    def sigmoid_prime(self, z):

        return(self.sigmoid(z)*(1 - self.sigmoid(z)))

    # Create mini batch of given size 
    # Use this for stochastic gradient descent
    # Mini-batches save time by learning from a sample
    # of training examples at each gradient descent iteration 
    def get_mini_batch(self):
        
        pass

    # Gradient descent
    # This function updates weights according to 
    # stochastic gradient descent formula given a 
    # learning rate and current gradient vector 
    # Create the update rule
    # Apply the update rule to update the weights
    # Then return the weights  
    # Input: Weights, learning rate, current gradient
    # Output: Updated weights from 1 iteration of gradient descent
    def gradient_descent(self, weights, rate, dW):

        return([w - rate*dW['dW' + str(layer)] 
                for weight in layer 
                for layer in weights])

    # Input: -
    # Output: Neural network training progress for every 100 SGD iterations
    def train(self):

        w, b = self.initialize_weights()
        
        for i in self.iterations:
            
            activations = forward_propagation(self.train, w, b)
            gradients = backward_propogation(gradients)
            weights = gradient_descent(weights, self.learning_rate, gradients)
            
            train_predictions = 
            test_predictions =  
            
            train_cost = get_cost(predictions)
            test_cost = get_cost(predictions)

            if iteration % 100 == 0:
                
                print("The training cost is: " + str(cost))
                print("The test cost is: " + str(cost))

                print("The training accuracy is: " + str(accuracy))
                print("The test accuracy is: " + str(accuracy))


# Now we will execute the script
# We will load the data-set 
# Next we will initialize a neural net object
# Then we will train the neural network 
# Then we will make predictions
if __name__ == main():
    
    # Set the parameters here
    data = train_test_split.TrainTestSplit()
    train = 
    test = 
    
    # Set the architecture 
    # Set the lLearning rate 
    # 
    arch = [4,3,2,1]
    rate = 0.6
    
    # Create a neural net object
    # Train this object 
    nn = NeuralNet(train, test, arch, rate)
    nn.train()


