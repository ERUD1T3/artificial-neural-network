############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 3/16/2022
#   file: testTennis.py
#   Description: main file to run the program for Iris dataset 
#   testIris: output accuracy on training and test sets.
#############################################################

# imports
from ann import ANN
from utils import Data


def main():
    '''main of the program'''

    weights_path = 'models/iris_weights.txt'
    debugging = False

    # hyperparameters# create data manager
    manager = Data(
        'data/iris/iris-train.txt',
        'data/iris/iris-test.txt', 
        'data/iris/iris-attr.txt',
        debugging
    )
    
    # hyperparameters
    h = {
            'k_fold': 0,
            'learning_rate': .01,
            'momentum': 0.0,
            'epochs': 5000,
            'decay': 0.0,
            'hidden_units': [ 6 ] # list of number of nodes in each layer
        }

    


    print('\nCreating NN with with parameters provided\n')
    # create the artificial neural network
    net = ANN(
        hyperparams=h, 
        input_units=manager.input_units,
        output_units=manager.output_units,
        debug=debugging
    )
    # printing the neural network
    net.print_network()


    print('\nLearning the NN...\n')
    # train the artificial neural network
    net.train(manager.training, manager.validation)
    print('\nTraining complete\n')

    #print weights
    print('\nPrinting learned weights\n')
    net.print_weights()

    # save the weights
    if weights_path:
        net.save(weights_path)
        print('weights saved to', weights_path)

    # test the artificial neural network
    print('\nTesting the NN on training set ...\n')
    accuracy = net.test(manager.training) * 100
    print('\nTraining set accuracy:', accuracy)

    # test the artificial neural network
    print('\nTesting the NN on testing set ...\n')
    accuracy = net.test(manager.testing) * 100
    print('\nTesting set accuracy:', accuracy)

    print('\nTesting complete\n')


    
if __name__ == '__main__':
    main()
