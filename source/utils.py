############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/23/2022
#   file: utils.py
#   Description: utility functions for artificial neural
#   network learning
#############################################################

import random

class Data:
    '''class to process data set
    '''
    def __init__(
        self, 
        training, 
        testing, 
        attributes, 
        debug):
        '''
        Initialize the APBT class
        '''
        self.debug = debug
        # reading attributes 
        self.attributes, self.in_attr, self.out_attr = self.read_attributes(attributes) 
        # reading input,output lenght
        self.input_units, self.output_units = self.get_input_output_len()
        # reading data
        if testing is None:
            self.training = self.read_data(training)
            self.testing = self.training
            self.validation = self.training
        else:
            self.training = self.read_data(training)
            self.testing = self.read_data(testing)
            self.n_examples = len(self.training)
            # suffle training data
            random.shuffle(self.training)
            # setting validation to 20% of the training data
            self.validation = self.training[:int(self.n_examples * 0.2)]
            self.training = self.training[int(self.n_examples * 0.2):]

        if self.debug:
            print('Training:', self.training)
            print('validation:', self.validation)
            print('Testing:', self.testing)

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
        # encode the values
        if len(values) > 1:
            if values[0] == '0' and values[1] == '1':
                return False
            else:
                return True
        else:
            return False

    def onehot(self, attr, value):
        '''
        Preprocess to convert a data instance 
        to one-of-n/onehot encoding
        '''
        # get the index of the value
        encoded = [0.0 for _ in range(len(self.attributes[attr]))]
        encoded[self.attributes[attr].index(value)] = 1.0

        return encoded

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
                    data.append([In, Out])

        if len(data) == 0:
            raise Exception('No data found')

        return data

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

    def get_input_output_len(self):
        '''
        Get the input and output units
        '''
        # getting total number of input units
        input_units = 0
        for attr in self.in_attr:
            values = self.attributes[attr]
            # check specifically for identity
            if values[0] == '0' and values[1] == '1':
                input_units += 1
            else:
                input_units += len(values)

        # getting total number of output units  
        output_units = 0
        for attr in self.out_attr:
            values = self.attributes[attr]
            # check specifically for identity
            if values[0] == '0' and values[1] == '1':
                output_units += 1
            else:
                output_units += len(values)

        return input_units, output_units

def log_csv(path, histories, headers):
        '''log the data to the csv file'''
        headers = ['e'] + headers
        # open the file
        with open(path, 'w') as f:
            # write the headers
            f.write(','.join(headers) + '\n')
            # write the data
            for h in range(len(histories[0])):
                line = f'{h},'
                for hh in range(len(histories)):
                    line += str(histories[hh][h]) + ','
                f.write(line[:-1] + '\n')

def corrupt_data(data, classes, percent):
    '''corrupt the class labels of training examples from 0% to 20% (2% in-
    crement) by changing from the correct class to another class; output the
    accuracy on the uncorrupted test set with and without rule post-pruning.'''

    # get the number of training examples
    num_examples = len(data)
    # get the number of classes to corrupt
    num_examples_to_corrupt = int(percent * num_examples)
    # get the elements to corrupt
    corrupt_elements = random.sample(range(num_examples), num_examples_to_corrupt)

    # corrupt the data
    for e in corrupt_elements:
        # get the class label
        correct_label = data[e][-1]
        
        random_class = random.choice(classes)

        # while the random class is the same as the correct class
        while random_class == correct_label:
            random_class = random.choice(classes)
    
        # change the class label
        data[e][-1] = random_class
        
    return data

    