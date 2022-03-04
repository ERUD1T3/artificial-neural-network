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
import math

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
        
        # hyperparameters
        self.hidden_units = hidden_units
        self.lr = lr
        self.momentum = momentum
        self.debug = debug
        self.epochs = epochs
        self.INIT_VAL = 0.01 # initial value for weights and biases
    
        # reading attributes 
        self.attributes, self.in_attr, self.out_attr = read_attributes(attributes, self.debug) 

        self.input_unit = len(self.in_attr)
        self.output_unit = len(self.out_attr)
        self.topology = None # ideally dynamically generated

        # reading data
        self.training = read_data(
            training, 
            self.input_unit, 
            self.output_unit, 
            self.debug
        )
        self.testing = read_data(
            testing, 
            self.input_unit, 
            self.output_unit, 
            self.debug
        )

        # case of discrete attributes
        if len(self.out_attr) == 1 \
            and len(self.attributes[self.out_attr[0]]) > 1:
            self.output_unit = len(self.attributes[self.out_attr[0]])


        # initialize the weights
        self.weights = {
            'hidden': [[self.INIT_VAL for _ in range(self.input_unit + 1)]
                        for _ in range(self.hidden_units)],
            'output': [[self.INIT_VAL for _ in range(self.hidden_units + 1)]
                        for _ in range(self.output_unit)]
        }

        # print the everything
        if self.debug:
            print('Training data: ', self.training)
            print('Testing data: ', self.testing)
            print('Attributes: ', self.attributes)
            print('Input attributes: ', self.in_attr)
            print('Output attributes: ', self.out_attr)
            print('learning rate: ', self.lr)
            print('momentum: ', self.momentum)
            print('epochs: ', self.epochs)
            print('Weights: ', self.weights)
            print('Input units: ', self.input_unit)
            print('Output units: ', self.output_unit)
            print('Hidden units: ', self.hidden_units)
             
    def print_weights(self):
        '''
        Print the weights of the Artificial Neural Network
        '''
        print('Weights: ', self.weights)

    def print_topology(self):
        '''
        Print the topology of the Artificial Neural Network
        '''
         # network topology
        self.topology = {
            'linear1': f'fully connected ({self.input_unit}x{self.hidden_units})',
            'activation1': 'sigmoid',
            'linear2': f'fully connected ({self.hidden_units}x{self.output_unit})',
            'activation2': 'sigmoid'
        }
        print('Topology: ', self.topology)

    def oneofn(self, attribute, values, _debug=False):
        '''
        Preprocess the data to one-of-n encoding
        '''

        # get the number of classes
        classes = len(values)

    def sigmoid(self, x):
        '''
        Sigmoid activation function
        '''
        return 1 / (1 + math.exp(-x))

    def d_sigmoid(self, x):
        '''
        Derivative of the sigmoid function
        '''
        y = self.sigmoid(x)
        return y * (1 - y)

    # TODO: test this function
    def feed_forward(self, instance):
        '''
        Feed forward the Artificial Neural Network
        '''
        hidden_res = [0.0 for _ in range(self.hidden_units)]
        output_res = [0.0 for _ in range(self.output_unit)]

        # feed forward the hidden layer
        for i in range(self.hidden_units):
            for j in range(self.input_unit):
                hidden_res[i] += self.weights['hidden'][i][j] * instance[j]
            hidden_res[i] += self.weights['hidden'][i][self.input_unit] # bias

        hidden_res = [self.sigmoid(x) for x in hidden_res]

        # feed forward the output layer
        for i in range(self.output_unit):
            for j in range(self.hidden_units):
                output_res[i] += self.weights['output'][i][j] * hidden_res[j]
            output_res[i] += self.weights['output'][i][self.hidden_units] # bias
    
        output_res = [self.sigmoid(x) for x in output_res]

        return output_res

    def back_propagate(self, targets):
        '''
        Back propagate the Artificial Neural Network
        with momentum and learning rate
        '''
        pass

    def train(self):
        '''
        Train the Artificial Neural Network
        '''
        pass


    def test(self, test_data=None):
        '''
        Test the Artificial Neural Network
        '''
        # null check
        test_data = test_data or self.testing
        
        if self.debug:
            print('Testing data: ', test_data)

    def save(self, filename):
        '''
        Save the Artificial Neural Network
        '''
        pass

    def load(self, filename):
        '''
        Load the Artificial Neural Network
        '''
        pass