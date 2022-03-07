############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/23/2022
#   file: main.py
#   Description: main file to run the program
#############################################################

# imports
import argparse
from ann import ANN

def parse_args():
    '''parse the arguments for artificial neural network'''

    parser = argparse.ArgumentParser(
        description='Artificial Neural Network for classification'
    )

    parser.add_argument(
        '-a', '--attributes',
        type=str,
        required=True,
        help='path to the attributes files (required)'
    )

    parser.add_argument(
        '-d', '--training',
        type=str, 
        required=True,
        help='path to the training data files (required)'
    )
    
    parser.add_argument(
        '-t', '--testing',
        type=str , 
        required=True,
        help='path to the test data files (required)'
    )

    parser.add_argument(
        '-w', '--weights',
        type=str , 
        required=False,
        help='path to save the weights (optional)'
    )

    parser.add_argument(
        '-u', '--hidden-units',
        type=int, 
        required=False,
        help='number of hidden units (default: 3)'
    )

    parser.add_argument(
        '-e', '--epochs',
        type=int, 
        required=False,
        default=10,
        help='number of epochs (default: 10)'
    )

    parser.add_argument(
        '-l', '--learning-rate',
        type=float, 
        required=False,
        default=0.1,
        help='learning rate (default: 0.1)',
    )

    parser.add_argument(
        '-m', '--momentum',
        type=float, 
        required=False,
        default=0.0,
        help='momentum (default: 0.0)',
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='debug mode, prints statements activated (optional)'
    )

    # parse arguments
    args = parser.parse_args()
    return args


def main():
    '''main of the program'''
    args = parse_args() # parse arguments
    print(' args entered',args)

    training_path = args.training
    testing_path = args.testing
    attributes_path = args.attributes
    weights_path = args.weights
    debugging = args.debug
    
    # hyperparameters
    hidden_units = args.hidden_units
    epochs = args.epochs
    lr = args.learning_rate
    momentum = args.momentum


    # create the artificial neural network
    ann = ANN(
        training_path,
        testing_path,
        attributes_path,
        hidden_units,
        lr,
        epochs,
        momentum,
        debugging
    )

    # print the network
    ann.print_network()

    # train the artificial neural network
    ann.train()

    # test the artificial neural network
    ann.test()
    
    # save the weights
    if weights_path:
        ann.save(weights_path)
        print('weights saved to', weights_path)
        # load the weights
        ann.load(weights_path)
        print('weights loaded from', weights_path)


    # feed forward
    inferance = ann.feed_forward(ann.training[0][0])
    print('inference', inferance)
    # decode the output
    print('decoded output', ann.decode(ann.attribute[ann.out_attr[0]],inferance))



    
if __name__ == '__main__':
    main()
