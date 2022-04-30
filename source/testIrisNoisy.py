############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 3/16/2022
#   file: testTennis.py
#   Description: main file to run the program for Iris dataset 
#   testIrisNoisy: corrupt 0% to 20% of class labels, with 2% 
#   increment, in the training set (similar to HW2); for each
#   level of noise, output accuracy on the uncorrupted test set;
#   use a validation set and not use a validation set (optionally 
#   use weight decay)
#############################################################

from ann import ANN
from utils import Data, corrupt_data

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

    
    # start of the experiment 
    for p in range(0, 22, 2):
        print('////////////////////////////////////////////////////////////')
        # corrupt the data'
        print('\nCorrupting the data by changing from the correct class to another class...')
        print('\nPercentage of corruption: {}%'.format(p))

        print('\nCreating NN without validation \n')
        # create the artificial neural network

        print('\nCreating NN with the parameters provided\n')
        # create the artificial neural network
        net = ANN(
        hyperparams=h, 
        input_units=manager.input_units,
        output_units=manager.output_units,
        debug=debugging
        )      
        # printing the neural network
        net.print_network()

        net.training = corrupt_data(manager.training, net.get_classes(), p / 100.)
        
        print('\nLearning the NN...\n')
        # train the artificial neural network
        net.train(manager.training, manager.validation)
        print('\nTraining complete\n')

        # save the weights
        if weights_path:
            net.save(weights_path)
            print('weights saved to', weights_path)

        # test the artificial neural network
        print('\nTesting the NN on testing set ...\n')
        accuracy = net.test(manager.testing) * 100
        print('\nTesting set accuracy:', accuracy)

        print('\nTesting complete\n')


    
if __name__ == '__main__':
    main()
