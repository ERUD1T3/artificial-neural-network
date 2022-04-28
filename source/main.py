############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/23/2022
#   file: main.py
#   Description: main file to run the program
#############################################################

# imports
import argparse
from source.utils import Data
from utils import Utils
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
        '-k', '--k-fold',
        type=int,
        required=False,
        help='number of folds for k-fold cross validation, k=0 or k=1 for no validation'
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
        help='learning rate (default: 0.01)',
    )

    parser.add_argument(
        '-m', '--momentum',
        type=float, 
        required=False,
        default=0.0,
        help='momentum (default: 0.9)',
    )

    parser.add_argument(
        '-g','--decay',
        type=float, 
        required=False,
        default=0.0,
        help='weight decay gamma (default: 0.01)',
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

    # create data manager
    manager = Data(
        args.training, 
        args.testing, 
        args.attributes, 
        args.debug)
    
    # hyperparameters
    h = {
            'k_fold': args.k_fold,
            'learning_rate': args.learning_rate,
            'momentum': args.momentum,
            'epochs': args.epochs,
            'decay': args.decay,
            'hidden_units': [ args.hidden_units ] # list of number of nodes in each layer
        }


    print('\nCreating NN with the parameters provided\n')
    # create the artificial neural network
    net = ANN(
        hyperparams=h, 
        input_units=manager.input_units,
        output_units=manager.output_units,
        debug=args.debug
    )

    # printing the neural network
    net.print_network()


    print('\nLearning the NN...\n')
    # train the artificial neural network
    net.train()
    print('\nTraining complete\n')

    #print weights
    print('\nPrinting learned weights\n')
    ann.print_weights()

    # save the weights
    if weights_path:
        ann.save(weights_path)
        print('weights saved to', weights_path)
        # load the weights
        # ann.load(weights_path)
        # print('weights loaded from', weights_path)

    # test the artificial neural network
    print('\nTesting the NN...\n')
    ann.test()
    print('\nTesting complete\n')


    
if __name__ == '__main__':
    main()
