#--------------------------------#
# Author: Akshat Goel
# Date: 28-Oct-2018
# Description: Basic Neural Net
#--------------------------------#

# Import libraries
import numpy as np 
import random 
import os 

# Import data
from sklearn.datasets import load_boston
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

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
    def __init__(self, X_train, y_train, X_test, y_test, arch, rate, iterations): 

        # Separate training labels
        # Separate training features
        # This assumes that the labels are the last column
        self.train_labels = y_train
        self.train_features = X_train

        # Separate test labels
        # Separate test features
        # This is the same as above but for the test data
        self.test_labels = X_test
        self.test_features = y_test
        
        # Store training data dimensions
        # n_x: No. of observations
        # n_y: No. of observations 
        # n_m: No. of features
        self.n_x = len(X_train)
        self.n_y = len(y_train)
        self.n_m = len(X_train[0])
        
        # Need to insert dimension of input layer
        # User inputs only architecture of hidden layers
        # Calculate the no. of layers only after this is done
        # This is so that we include the input layer in the total layers
        self.arch = arch
        self.layers = range(1, len(arch))
        
        # Store the learning rate
        # Store the number of iterations
        self.rate = rate
        self.iterations = iterations
    
    # Input: Vector with no. of neurons in each layer
    # Output: Vector of initialized weights and biases
    def get_initial_weights(self):
        
        # This contains data for:
        # Architecture
        # Layers 

        arch = self.arch
        layers = self.layers

        print(self.arch)
        print(arch)
        print(layers)

        # Initialize weights according to a standard normal distribution
        # Store initialization in a dictionary where the key is the layer number
        W = {l: np.random.randn(arch[l], arch[l - 1]) for l in layers}
        b = {l: np.random.randn(arch[l], 1) for l in layers} 
        
        return(W, b)

    # Forward propagation:
    # Calculate the activations for each neuron
    # The activation function is given later below 
    # Each neuron either fires or doesn't
    def forward_propogation(self, X, Y, W, B):

        # Store the layers 
        # iterable
        layers = self.layers 
        L = layers[-1]

        # This is the initial layer
        # where to store information 
        A = {0: X}
        Z = {1: X}
        
        # Now iterate over layers
        for l in layers:

        	Z[l] = np.add(W[l] @ A[l-1], B[l])
        	A[l] = self.sigmoid(Z[l])

        # Store the 'error' term from the last layer
        dZL = self.sigmoid_prime(Z[L]) * (A[L] - Y)
                
        return(dZL, A, W, B, Z) 

    # Backward propogation
    # Propagate the output layer's error backward
    # Calculations are given by back-propagation equations
    # This allows us to see how cost is impacted by adjusting weights in previous layers 
    def backward_propogation(self, dZL, A, W, b, Z):

        # Initalize dictionaries to store the derivatives
        L = len(self.arch) - 1
        
        dZ = {L: dZL}
        dW = {}
        db = {}
        
        for l in reversed(range(1, L + 1)):

            dZ[l-1] = (W[l].T @ dZ[l]) * self.sigmoid_prime(A[l-1]) 
            dW[l] = dZ[l] @ A[l-1].T
            db[l] = dZ[l]

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

    # Gradient descent
    # This function updates weights according to 
    # stochastic gradient descent formula given a 
    # learning rate and current gradient vector 
    # Create the update rule
    # Apply the update rule to update the weights
    # Then return the weights  
    # Input: Weights, learning rate, current gradient
    # Output: Updated weights from 1 iteration of gradient descent
    def gradient_descent(self, W, b, dW, db, rate):

    	W = {l: np.subtract(W[l], rate*dW[l]) for l in W.keys()}
    	b = {l: np.subtract(b[l], rate*db[l]) for l in b.keys()}

    	return(W, b)

    # Input: Training and test data-sets
    # Output: Neural network training progress for every 100 SGD iterations
    def train(self):


        w, b = self.get_initial_weights()
        L = len(self.arch) - 1

        for iteration in range(self.iterations):
            
            dZL, A, W, b, Z = self.forward_propogation(self.train_features, self.train_labels, w, b)
            dW, db = self.backward_propogation(dZL, A, W, b, Z)
            W, b = self.gradient_descent(W, b, dW, db, self.rate)
                        
            train_preds = A[L]
            train_cost = self.get_cost(train_preds, self.train_labels)
           
            if iteration % 100 == 0:
                print("The training cost is: " + str(train_cost))


# Input: Load and process data
def load_data(path, x_name, y_name, test_size, random_state):
    
    df = pd.read_csv(path)
    X, y = np.array(df[x_name]), np.array(df[y_name])
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)

    # Prepare dimensions of test data
    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)
    
    # Perform transformations
    data = [X_train, X_test, y_train, y_test]
    results = [RobustScaler().fit(X).transform(X) for X in data]
 	
    # Return statement
    return(*results, )

                
# Now we will execute the script
# We will load the data-set 
# Next we will initialize a neural net object
# Then we will train the neural network 
# Then we will make predictions
if __name__ == '__main__':

    # Process data
    X_train, X_test, y_train, y_test = load_data(test_size=0.33, random_state=42)
    
    # Set the architecture
    # Add in the input layer
    # Set the learning rate 
    # Set the number of iterations
    arch = [len(X_train), 600, 575, 500, 450, 400, 350, len(y_train)]
    iterations = 50000
    rate = 0.6

    
    # Create a neural net object
    # Train this object 
    nn = NeuralNet(X_train, y_train, X_test, y_test, arch, rate, iterations)
    nn.train()
