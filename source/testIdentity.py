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


def main():
    '''main of the program'''

    training_path = 'data/identity/itentity-train.txt'
    testing_path = 'data/identity/identity-test.txt'
    attributes_path = 'data/identity/identity-attr.txt'
    weights_path = 'identity_weights.txt'
    debugging = False
    
    # hyperparameters
    epochs = 5000
    learning_rate = 1e-3
    decay = 0.001
    momentum = 0.09
    k_folds = 3

    hidden_units = 3


    print('\nCreating NN with with 3 hidden units\n')
    # create the artificial neural network
    ann3 = ANN(
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
    ann3.print_network()


    print('\nLearning the NN...\n')
    # train the artificial neural network
    ann3.train()
    print('\nTraining complete\n')

    #print weights
    print('\nPrinting learned weights\n')
    ann3.print_weights()

    # save the weights
    if weights_path:
        ann3.save(weights_path)
        print('weights saved to', weights_path)
        # load the weights
        # ann.load(weights_path)
        # print('weights loaded from', weights_path)

    # test the artificial neural network
    print('\nTesting the NN on training set ...\n')
    accuracy = ann3.test(ann3.training) * 100
    print('\nTraining set accuracy:', accuracy)

    # test the artificial neural network
    print('\nTesting the NN on testing set ...\n')
    accuracy = ann3.test(ann3.testing) * 100
    print('\nTesting set accuracy:', accuracy)

    print('\nTesting complete\n')


    hidden_units = 4


    print('\nCreating NN with with 4 hidden units\n')
    # create the artificial neural network
    ann4 = ANN(
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
    ann4.print_network()


    print('\nLearning the NN...\n')
    # train the artificial neural network
    ann4.train()
    print('\nTraining complete\n')

    #print weights
    print('\nPrinting learned weights\n')
    ann4.print_weights()

    # save the weights
    if weights_path:
        ann4.save(weights_path)
        print('weights saved to', weights_path)
        # load the weights
        # ann.load(weights_path)
        # print('weights loaded from', weights_path)

    # test the artificial neural network
    print('\nTesting the NN on training set ...\n')
    accuracy = ann4.test(ann4.training) * 100
    print('\nTraining set accuracy:', accuracy)

    # test the artificial neural network
    print('\nTesting the NN on testing set ...\n')
    accuracy = ann4.test(ann4.testing) * 100
    print('\nTesting set accuracy:', accuracy)

    print('\nTesting complete\n')


    
if __name__ == '__main__':
    main()
