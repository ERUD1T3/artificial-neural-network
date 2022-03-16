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
        validation,
        weights_path,
        hidden_units, 
        learning_rate, 
        epochs,
        momentum, 
        decay,
        debug=True,
    ) -> None:
        
        '''
        Initialize the Artificial Neural Network
        '''
        self.topology = None # ideally dynamically generated
        self.validation = []
        # hyperparameters
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.debug = debug
        self.epochs = epochs
        self.INIT_VAL = 0.0 # initial value for weights and biases
        self.weights_path = weights_path
    
        # reading attributes 
        self.attributes, self.in_attr, self.out_attr = self.read_attributes(attributes) 

        # getting total number of input units
        self.input_units = 0
        for attr in self.in_attr:
            values = self.attributes[attr]
            # check specifically for identity
            if values[0] == '0' and values[1] == '1':
                self.input_units += 1
            else:
                self.input_units += len(values)

        # getting total number of output units  
        self.output_units = 0
        for attr in self.out_attr:
            values = self.attributes[attr]
            # check specifically for identity
            if values[0] == '0' and values[1] == '1':
                self.output_units += 1
            else:
                self.output_units += len(values)
       
        # reading data
        self.training = self.read_data(training)
        self.testing = self.read_data(testing)
        self.n_examples = len(self.training)

        # if validation is true, then we will use the validation set
        if validation:
            # extract 20% of training set for validation
            cutoff = int(self.n_examples * 0.30) # sensistive to size of validation set
            self.validation = self.training[:cutoff]
            self.training = self.training[cutoff:]
            self.n_examples = len(self.training)

        # case of discrete attributes
        # if len(self.out_attr) == 1 \
        #     and len(self.attributes[self.out_attr[0]]) > 1:
        #     self.output_units = len(self.attributes[self.out_attr[0]])


        # initialize the weights
        self.weights = {
            'hidden': [[self.INIT_VAL for _ in range(self.input_units + 1)]
                        for _ in range(self.hidden_units)],
            'output': [[self.INIT_VAL for _ in range(self.hidden_units + 1)]
                        for _ in range(self.output_units)]
        }

        # print the everything
        if self.debug:
            print('Training data: ', self.training)
            print('Testing data: ', self.testing)
            print('Attributes: ', self.attributes)
            print('Input attributes: ', self.in_attr)
            print('Output attributes: ', self.out_attr)
            print('learning rate: ', self.learning_rate)
            print('momentum: ', self.momentum)
            print('epochs: ', self.epochs)
            print('Weights: ', self.weights)
            print('Input units: ', self.input_units)
            print('Output units: ', self.output_units)
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
            'linear1': f'fully connected ({self.input_units}x{self.hidden_units})',
            'activation1': 'sigmoid',
            'linear2': f'fully connected ({self.hidden_units}x{self.output_units})',
            'activation2': 'sigmoid'
        }
        print('Topology: ', self.topology)

    def print_network(self):
        '''
        Print the network
        '''
        self.print_topology()
        self.print_weights()

    def save(self, filename=None):
        '''
        Save the Artificial Neural Network
        '''
        # if no filename is provided, then use the default
        if filename is None:
            filename = self.weights_path
        # save the weights onto a file
        with open(filename, 'w') as f:
            f.write(str(self.weights))

    def load(self, filename):
        '''
        Load the Artificial Neural Network
        '''
        # load the weights from a file
        with open(filename, 'r') as f:
            self.weights = eval(f.read())

        # print('one weight: ', self.weights['hidden'][0][0])

        self.print_weights()


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

    def to_encode(self, attr):
        '''
        Return true if the value is discrete
        to encode
        '''
        values = self.attributes[attr]

        # if self.debug:
        #     print('values: ', values)
        #     print('the attribute to encode is: ', attr)

        if len(values) > 1:
            if values[0] == '0' and values[1] == '1':
                return False
            else:
                return True
        else:
            return False

    def read_data(self, data_path):
        '''
        Read in the training data and testing data
        '''
        data = []

        # read in the attributes
        with open(data_path, 'r') as f:
            for line in f:
                if len(line) > 0:
                    items = line.strip().split()

                    # if self.debug:
                    #     print('Items: ', items)

                    # get items iterator
                    items_iter = iter(items)

                    In, Out = [],[]
                    # get inputs
                    for attr in self.in_attr:
                        value = next(items_iter)
                        if self.to_encode(attr):
                            # encode discrete values
                            encoded = self.onehot(attr, value)
                            In += encoded # since encoded is a list
                        else:
                            # encode continuous values
                            In.append(float(value))

                    # get outputs
                    for attr in self.out_attr:
                        value = next(items_iter)
                        if self.to_encode(attr):
                            # encode discrete values
                            encoded = self.onehot(attr, value)
                            Out += encoded # since encoded is a list
                        else:
                            # encode continuous values
                            Out.append(float(value))

                    # check if the encoding should be applied
                    # when encoding applied, update the input or output units sizes

                    data.append((In, Out))
                    
                    
        # if self.debug:
        #     print('Read data: ', data)

        if len(data) == 0:
            raise Exception('No data found')

        return data


    def onehot(self, attr, value):
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
    

        # get the index of the value
        encoded = [0.0 for _ in range(len(self.attributes[attr]))]
        encoded[self.attributes[attr].index(value)] = 1.0

        if self.debug:
            print('One-hot encoded: ', encoded)

        return encoded

    def decode(self, attr, encoded):
        '''
        Decode the encoded value
        '''
        # get the index of the value
        # value = self.attributes[attr][encoded.index(1.0)]
        if self.debug:
            print('Encoded: ', encoded)
            print('attr: ', attr)
            print('Attributes: ', self.attributes[attr])
        value_encoded = zip(self.attributes[attr], encoded)
        # sort the encoded value
        sorted_encoded = sorted(value_encoded, key=lambda x: x[1], reverse=True)

        # get the value
        value = sorted_encoded[0][0]

        if self.debug:
            print('Decoded: ', value)
            print('Sorted encoded: ', sorted_encoded)

        return value

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

    def feed_forward(self, instance):
        '''
        Feed forward the Artificial Neural Network
        '''
        hidden_res = [0.0 for _ in range(self.hidden_units)]
        output_res = [0.0 for _ in range(self.output_units)]

        # feed forward the hidden layer
        for i in range(self.hidden_units):
            for j in range(self.input_units):
                hidden_res[i] += self.weights['hidden'][i][j] * instance[j]
            hidden_res[i] += self.weights['hidden'][i][self.input_units] # bias

        self.hidden_res = [self.sigmoid(x) for x in hidden_res]

        # feed forward the output layer
        for i in range(self.output_units):
            for j in range(self.hidden_units):
                output_res[i] += self.weights['output'][i][j] * self.hidden_res[j]
            output_res[i] += self.weights['output'][i][self.hidden_units] # bias
    
        output_res = [self.sigmoid(x) for x in output_res]

        return output_res

    def predict(self, line_instance):
        '''
        Predict the output of the instance

        '''

        # items = line_instance.strip().split()

        # if self.debug:
        #     print('Items: ', items)

        # get items iterator
        # items_iter = iter(items)

        # instance = []
        # # get inputs
        # for attr in self.in_attr:
        #     value = next(items_iter)
        #     if self.to_encode(attr):
        #         # encode discrete values
        #         encoded = self.onehot(attr, value)
        #         instance += encoded # since encoded is a list
        #     else:
        #         # encode continuous values
        #         instance.append(float(value))

        # feed forward the network
        # output_res = self.feed_forward(instance)

        # get the index of the max value
        # max_index = output_res.index(max(output_res))

        # # get the value of the max value
        # for attr in self.out_attr:

        # max_value = self.decode(self.out_attr[max_index], output_res)

        # return max_value

        return self.feed_forward(line_instance)

    def loss(self, target, output):
        '''
        Compute the loss for SGD
        '''
        loss = 0.0
        # getting all the loss
        for i in range(self.output_units):
            loss += (target[i] - output[i]) ** 2
        loss /= 2.0

        weights_term = 0.0
        # adding all the weights
        for i in range(self.hidden_units):
            for j in range(self.input_units + 1):
                weights_term += self.weights['hidden'][i][j] ** 2
        for i in range(self.output_units):
            for j in range(self.hidden_units + 1):
                weights_term += self.weights['output'][i][j] ** 2

        loss += self.decay * weights_term

        return loss

    def back_propagate(self, instance, output):
        '''
        Back propagate the error with momentum, weight 
        decay and learning rate
        SGD
        '''

        # prior delta update
        deltas = {
            'hidden': [[0.0 for _ in range(self.input_units + 1)]
                        for _ in range(self.hidden_units)],
            'output': [[0.0 for _ in range(self.hidden_units + 1)]
                        for _ in range(self.output_units)]
        }

        # get the target 
        target = instance[1]
        inputs = instance[0]

        if self.debug:
            print('inputs: ', inputs)
            print('Target: ', target)
            print('Output: ', output)

        # compute the error for output layer
        error = [0.0 for _ in range(self.output_units)]
        for i in range(self.output_units):
            error[i] = (target[i] - output[i]) * self.d_sigmoid(output[i]) 

        # compute the error for hidden layer
        hidden_error = [0.0 for _ in range(self.hidden_units)]
        for i in range(self.hidden_units):
            for j in range(self.output_units):
                hidden_error[i] += error[j] * self.weights['output'][j][i]
            hidden_error[i] *= self.d_sigmoid(self.hidden_res[i])


        # weight update factor based on decay
        factor = 1 - 2 * self.learning_rate * self.decay

        # update the weights
        for i in range(self.output_units):
            for j in range(self.hidden_units):
                deltas['output'][i][j] = factor * error[i] * self.hidden_res[j] \
                                + self.momentum * deltas['output'][i][j]
                self.weights['output'][i][j] += deltas['output'][i][j]
            
            deltas['output'][i][self.hidden_units] = factor * error[i] \
                                + self.momentum * deltas['output'][i][self.hidden_units]
            self.weights['output'][i][self.hidden_units] += deltas['output'][i][self.hidden_units]

        for i in range(self.hidden_units):
            for j in range(self.input_units):
                deltas['hidden'][i][j] = factor * hidden_error[i] * inputs[j] \
                                + self.momentum * deltas['hidden'][i][j]
                self.weights['hidden'][i][j] += deltas['hidden'][i][j]
            deltas['hidden'][i][self.input_units] = factor * hidden_error[i] \
                                + self.momentum * deltas['hidden'][i][self.input_units]  
            self.weights['hidden'][i][self.input_units] += deltas['hidden'][i][self.input_units]



    # TODO: add k-fold cross validation
    def train(self):
        '''
        Train the Artificial Neural Network
        add
        '''

        # get the data
        data = self.training
        # shuffle the data
        loss = 0.0
        # get validation data
        validation = self.validation

        # train the network
        for i in range(self.epochs):

            if self.debug:
                print('Epoch: ', i)

            for instance in data:
                output = self.feed_forward(instance[0])
                self.back_propagate(instance, output)
                loss += self.loss(instance[1], output)
                
               
                
                # compute the validation loss
                # validation_loss = 0.0
                # for instance in validation:
                #     output = self.feed_forward(instance[0])
                #     validation_loss += self.loss(instance[1], output)

                # # if self.debug:
                # #     print('Validation Loss: ', validation_loss/len(validation))

                # # check if the loss is decreasing
                # if validation_loss > loss:
                #     break

            if self.debug:
                # print('Weights: ', self.weights)
                print('Loss: ', loss/len(data))
           
            # if loss/len(data) < 1:
            #     break
            
            # if loss/len(data) < 5:
            #     break
            

        # save the weights
        # self.save()


    def test(self, test_data=None):
        '''
        Test the Artificial Neural Network
        '''
        # get the data, null check  
        test_data = test_data or self.testing
        
        if self.debug:
            print('Testing data: ', test_data)

        accuracy = 0.0

        # test the network
        for instance in test_data:
            output = self.feed_forward(instance[0])
            print('Output: ', output)
            print('Target: ', instance[1])
            print('Loss: ', self.loss(instance[1], output))

            # check check how correct is the output
            # if output == instance[1]:
            #     accuracy += 1.0
            accuracy += 1.0 - self.loss(instance[1], output)

        accuracy /= len(test_data)

        if self.debug:
            print('Accuracy: ', accuracy * 100, '%')
        
        return accuracy
 



        

