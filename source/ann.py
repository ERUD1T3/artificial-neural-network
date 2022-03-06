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
        self.attributes, self.in_attr, self.out_attr = self.read_attributes(attributes) 

        # getting total number of input units
        self.input_unit = 0
        for attr in self.in_attr:
            values = self.attributes[attr]
            # check specifically for identity
            if values[0] == '0' and values[1] == '1':
                self.input_unit += 1
            else:
                self.input_unit += len(values)

        # getting total number of output units  
        self.output_unit = 0
        for attr in self.out_attr:
            values = self.attributes[attr]
            # check specifically for identity
            if values[0] == '0' and values[1] == '1':
                self.output_unit += 1
            else:
                self.output_unit += len(values)
        self.topology = None # ideally dynamically generated

        # reading data
        self.training = self.read_data(
            training, 
            self.input_unit, 
            self.output_unit
        )
        self.testing = self.read_data(
            testing, 
            self.input_unit, 
            self.output_unit
        )

        # case of discrete attributes
        # if len(self.out_attr) == 1 \
        #     and len(self.attributes[self.out_attr[0]]) > 1:
        #     self.output_unit = len(self.attributes[self.out_attr[0]])


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

    # TODO: test this function
    def save(self, filename):
        '''
        Save the Artificial Neural Network
        '''
        # save the weights onto a file
        with open(filename, 'w') as f:
            f.write(str(self.weights))


    # TODO: test this function
    def load(self, filename):
        '''
        Load the Artificial Neural Network
        '''
        # load the weights from a file
        with open(filename, 'r') as f:
            self.weights = eval(f.read())


    def read_attributes(self, attr_path):
        '''
        Read in the attributes
        '''

        attributes = {}
        in_attr, out_attr = [], []
        is_input = True

            # read in the attributes
        with open(attr_path, 'r') as f:
            for line in f:
                if len(line) > 1:
                    words = line.strip().split()
                    
                    # storing the attributes
                    attributes[words[0]] = words[1:]

                    # storing the input attributes
                    if is_input:
                        in_attr.append(words[0])
                    else:
                        out_attr.append(words[0])
                    # order.append(words[0])
                else:
                    is_input = False

                
        if self.debug:
            print('Attributes: ', attributes)
            print('Input attributes: ', in_attr)
            print('Output attributes: ', out_attr)

        if len(attributes) == 0:
            raise Exception('No attributes found')


        return attributes, in_attr, out_attr


    def read_data(self, data_path, input_size, output_size):
        '''
        Read in the training data and testing data
        '''
        data = []
        In = [None for _ in range(input_size)]
        Out = [None for _ in range(output_size)]

        # read in the attributes
        with open(data_path, 'r') as f:
            for line in f:
                items = line.strip().split()

                # if self.debug:
                #     print('Items: ', items)

                # get items iterator
                items_iter = iter(items)

                # get inputs
                for i in range(input_size):
                    In[i] = (next(items_iter))

                # get outputs
                for o in range(output_size):
                    Out[o] = (next(items_iter))

                # check if the encoding should be applied
                # when encoding applied, update the input or output units sizes

                data.append(
                    self.onehot(
                        (In, Out), 
                        self.attributes, 
                        self.in_attr,
                        self.out_attr,
                    ))
                    
        if self.debug:
            print('Read data: ', data)

        if len(data) == 0:
            raise Exception('No data found')

        self.input_unit = 
        return data


    def onehot(self, instance, attr_values, in_attrs, out_attrs):
        '''
        Preprocess to convert a data instance 
        to one-of-n/onehot encoding
        '''
        #Input attributes is discrete
        # Outlook Sunny Overcast Rain -> Outlook: [a, b, c]
        # Temperature Hot Mild Cool -> Temperature: [d, e, f]
        # Humidity High Normal -> Humidity: [g, h]
        # Wind Weak Strong -> Wind: [i, j]
        # Concatenate all encoded attributes
        # [a, b, c, d, e, f, g, h, i, j]

        #Output attributes is discrete
        # PlayTennis Yes No -> PlayTennis [x,y]

        # input output pairs are 
        # ([a, b, c, d, e, f, g, h, i, j], [x,y]), ...]
        # return instance
        


        encoded = {
            attr: [0 for _ in range(len(attr_values[attr]))] 
            for attr in (in_attrs + out_attrs)
        } 

        # loop through input attributes
        for i, attr in enumerate(in_attrs):
            # get the index of the attribute value
            index = attr_values[attr].index(instance[0][i])
            # set the index to 1
            encoded[attr][index] = 1

        # loop through output attributes
        for o, attr in enumerate(out_attrs):
            # get the index of the attribute value
            index = attr_values[attr].index(instance[1][o])
            # set the index to 1
            encoded[attr][index] = 1

        if self.debug:
            print('One-hot encoded: ', encoded)

        In = []
        Out = []

        # clean up encoded
        for attr in in_attrs:
            In += encoded[attr]

        for attr in out_attrs:
            Out += encoded[attr]

        if self.debug:
            print('One-hot encoded: ', In, Out)

        return (In, Out)

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