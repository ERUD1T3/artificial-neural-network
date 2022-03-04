############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/23/2022
#   file: utils.py
#   Description: utility functions for artificial neural
#   network learning
#############################################################

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

            if _debug:
                print('Items: ', items)

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


def preprocess_oneofn(data, classes, _debug=False):
    '''
    Preprocess the data to one-of-n encoding
    '''

    # get the number of classes
    num_classes = len(classes)

    # get the number of training examples
    num_examples = len(data)

    # get the number of attributes
    num_attributes = len(data[0]) - 1

    # create the one-of-n encoding
    oneofn = []

    # create the one-of-n encoding
    for i in range(num_examples):
        oneofn.append([])
        for j in range(num_attributes):
            oneofn[i].append(0)

        # get the class label
        class_label = data[i][-1]

        # get the index of the class label
        class_index = classes.index(class_label)

        # set the one-of-n encoding
        oneofn[i][class_index] = 1

    if _debug:
        print('One-of-n encoding: ', oneofn)

    return oneofn