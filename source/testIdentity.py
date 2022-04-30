############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 3/16/2022
#   file: testIdentity.py
#   Description: main file to run the program for identity 
#   testIdentity: output accuracy on training set and hidden 
#   values (similar to the format in Figure 4.7)
#   for each input using 3 and 4 hidden units;
#   for hidden values (with 2 decimal places), add binary 
#   values using 0.5 as the threshold; for the
#   sample first row of Figure 4.7: 0.89 0.04 0.08 (1 0 0)
#   for output values, print the actual output values
#   (with 1 decimal place)
#############################################################

# imports
from ann import ANN
from utils import Data


def main():
    '''main of the program'''

    weights_path = 'identity_weights.txt'
    debugging = False
    
    # hyperparameters# create data manager
    manager = Data(
        'data/identity/identity-train.txt',
        None, 
        'data/identity/identity-attr.txt',
        debugging
    )
    
    # hyperparameters
    h1 = {
            'k_fold': 0,
            'learning_rate': .005,
            'momentum': 0.0,
            'epochs': 100000,
            'decay': 0.0,
            'hidden_units': [ 3 ] # list of number of nodes in each layer
        }

    print('\nCreating NN with with 3 hidden units\n')
    # create the artificial neural network
    # create the artificial neural network
    net1 = ANN(
        hyperparams=h1, 
        input_units=manager.input_units,
        output_units=manager.output_units,
        debug=debugging
    )
    # printing the neural network
    net1.print_network()


    print('\nLearning the NN...\n')
    # train the artificial neural network
    net1.train(manager.training, manager.validation)
    print('\nTraining complete\n')

    #print weights
    print('\nPrinting learned weights\n')
    net1.print_weights()



    # test the artificial neural network
    net1.debug = True
    print('\nTesting the NN on testing set ...\n')
    net1.test(manager.testing)


    h2 = {
            'k_fold': 0,
            'learning_rate': .005,
            'momentum': 0.0,
            'epochs': 100000,
            'decay': 0.0,
            'hidden_units': [ 4 ] # list of number of nodes in each layer
        }



    print('\nCreating NN with with 4 hidden units\n')
    # create the artificial neural network
    # create the artificial neural network
    net2 = ANN(
        hyperparams=h2, 
        input_units=manager.input_units,
        output_units=manager.output_units,
        debug=debugging
    )
    # printing the neural network
    net2.print_network()


    print('\nLearning the NN...\n')
    # train the artificial neural network
    net2.train(manager.training, manager.validation)
    print('\nTraining complete\n')

    #print weights
    print('\nPrinting learned weights\n')
    net2.print_weights()



    # test the artificial neural network
    net2.debug = True
    print('\nTesting the NN on testing set ...\n')
    net2.test(manager.testing)


    
if __name__ == '__main__':
    main()
