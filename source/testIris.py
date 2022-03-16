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


def main():
    '''main of the program'''


    training_path = 'data/iris/iris-train.txt'
    testing_path = 'data/iris/iris-test.txt'
    attributes_path = 'data/iris/iris-attr.txt'
    weights_path = 'iris_weights.txt'
    debugging = False
    
    # hyperparameters
    hidden_units = 6
    epochs = 5000
    learning_rate = .01
    decay = 0.0
    momentum = 0.0
    k_folds = 5


    print('\nCreating NN with the parameters provided\n')
    # create the artificial neural network
    ann = ANN(
        training_path, # path to training data
        testing_path, # path to testing data
        attributes_path, # path to attributes
        k_folds, # whether to use validation data
        weights_path, # path to save weights
        hidden_units, # number of hidden units
        learning_rate, # learning rate
        epochs, # number of epochs, -1 for stopping based on validation
        momentum, # momentum
        decay, # weight decay gamma
        debugging # whether to print debugging statements
    )
    # printing the neural network
    ann.print_network()


    print('\nLearning the NN...\n')
    # train the artificial neural network
    ann.train()
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
    print('\nTesting the NN on training set ...\n')
    accuracy = ann.test(ann.training) * 100
    print('\nTraining set accuracy:', accuracy)

    # test the artificial neural network
    print('\nTesting the NN on testing set ...\n')
    accuracy = ann.test(ann.testing) * 100
    print('\nTesting set accuracy:', accuracy)

    print('\nTesting complete\n')


    
if __name__ == '__main__':
    main()
