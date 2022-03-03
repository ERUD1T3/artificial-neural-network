############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/23/2022
#   file: ann.py
#   Description: main class for artificial neural network
#   Implementation of backpropagation algorithm for a 
#   feed forward neural network with one hidden layer
#   input parameters are:
#   - learning rate
#   - number of hidden units
#   - number of iterations
#   - momentum
#############################################################

from utils import read_attributes, read_data


class ANN:
    '''
    Feed Forward Artificial Neural Network Class
    1 Input, 1 Hidden, 1 Output Layer
    '''
    def __init__(
        self, 
        training,
        testing,
        attributes,
        hidden_units, 
        lr, 
        epochs,
        momentum, 
        debug=True
    ) -> None:
        
        '''
        Initialize the Artificial Neural Network
        '''

        # reading data and attributes 
        self.training = read_data(training, self.debug)
        self.testing = read_data(testing, self.debug)
        self.attributes, self.in_attr, self.out_attr = read_attributes(self.attributes, self.debug) 
        
        
        # hyperparameters
        self.hidden_units = hidden_units
        self.lr = lr
        self.momentum = momentum
        self.debug = debug
    

        # initialize the weights
        input_unit = len(self.in_attr)
        output_unit = len(self.out_attr)

        self.weights = [[
            [0.0 for _ in range(input_unit + 1)]
            for _ in range(self.hidden_units) 
        ]]

        

    def train(self, inputs, targets):
        '''
        Train the Artificial Neural Network
        '''
        pass