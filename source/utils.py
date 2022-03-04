############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/23/2022
#   file: utils.py
#   Description: utility functions for artificial neural
#   network learning
#############################################################

from base64 import encode
import random

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


def read_attributes(attr_path, _debug=False):
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

            
    if _debug:
        print('Attributes: ', attributes)
        print('Input attributes: ', in_attr)
        print('Output attributes: ', out_attr)

    if len(attributes) == 0:
        raise Exception('No attributes found')


    return attributes, in_attr, out_attr


def read_data(data_path, input_size, output_size, _debug=False):
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

            # if _debug:
            #     print('Items: ', items)

            # get items iterator
            items_iter = iter(items)

            # get inputs
            for i in range(input_size):
                In[i] = (next(items_iter))

            # get outputs
            for o in range(output_size):
                Out[o] = (next(items_iter))

            data.append((In, Out))
                
    if _debug:
        print('Read data: ', data)

    if len(data) == 0:
        raise Exception('No data found')

    return data


def onehot(instance, attr_values, in_attrs, out_attrs, _debug=False):
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
    # [([a, b, c, d, e, f, g, h, i, j], [x,y]), ...]

    # get the number of input attribute values 
    num_in_attr_values = 0
    for attr in in_attrs:
        num_in_attr_values += len(attr_values[attr])

    # get the number of output attribute values
    num_out_attr_values = 0
    for attr in out_attrs:
        num_out_attr_values += len(attr_values[attr])

    encoded_in = [0 for _ in range(num_in_attr_values)]
    encoded_out = [0 for _ in range(num_out_attr_values)]

    # loop through input attributes
    for a in len(in_attrs):
        attr = in_attrs[a]
        # get the index of the attribute value
        index = attr_values[attr].index(instance[attr])
        # set the index to 1
        encoded_in[a * index] = 1
