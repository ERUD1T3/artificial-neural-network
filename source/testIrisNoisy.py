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
from utils import corrupt_data

def main():
    '''main of the program'''

    training_path = 'data/iris/iris-train.txt'
    testing_path = 'data/iris/iris-test.txt'
    attributes_path = 'data/iris/iris-attr.txt'
    weights_path = 'iris_weights.txt'
    debugging = False
    
    # hyperparameters
    hidden_units = 6
    epochs = 500
    learning_rate = .01
    decay = 0.0
    momentum = 0.0
    
    # start of the experiment 
    for p in range(0, 22, 2):
        print('////////////////////////////////////////////////////////////')
        # corrupt the data'
        print('\nCorrupting the data by changing from the correct class to another class...')
        print('\nPercentage of corruption: {}%'.format(p))

        print('\nCreating NN without validation \n')
        # create the artificial neural network
        k_folds = 0

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

        ann.training = corrupt_data(ann.training, ann.get_classes(), p / 100.)
        
        print('\nLearning the NN...\n')
        # train the artificial neural network
        ann.train()
        print('\nTraining complete\n')

        #print weights
        # print('\nPrinting learned weights\n')
        # ann.print_weights()

        # save the weights
        if weights_path:
            ann.save(weights_path)
            print('weights saved to', weights_path)
            
            # load the weights
            # ann.load(weights_path)
            # print('weights loaded from', weights_path)

        # test the artificial neural network
        # print('\nTesting the NN on training set ...\n')
        # accuracy = ann.test(ann.training) * 100
        # print('\nTraining set accuracy:', accuracy)

        # test the artificial neural network
        print('\nTesting the NN on testing set ...\n')
        accuracy = ann.test(ann.testing) * 100
        print('\nTesting set accuracy:', accuracy)

        print('\nTesting complete\n')
######################################################################
        # create the artificial neural network
        print('\nCreating NN with k-fold validation\n')
        k_folds = 4

        print('\nCreating NN with the parameters provided\n')
        # create the artificial neural network
        ann_v = ANN(
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
        # ann_v.print_network()

        ann_v.training = ann.training
        
        print('\nLearning the NN...\n')
        # train the artificial neural network
        ann_v.train()
        print('\nTraining complete\n')

        #print weights
        # print('\nPrinting learned weights\n')
        # ann_v.print_weights()

        # save the weights
        if weights_path:
            ann_v.save(weights_path)
            print('weights saved to', weights_path)
            
            # load the weights
            # ann.load(weights_path)
            # print('weights loaded from', weights_path)

        # test the artificial neural network
        # print('\nTesting the NN on training set ...\n')
        # accuracy = ann.test(ann.training) * 100
        # print('\nTraining set accuracy:', accuracy)

        # test the artificial neural network
        print('\nTesting the NN on testing set ...\n')
        accuracy = ann_v.test(ann_v.testing) * 100
        print('\nTesting set accuracy:', accuracy)

        print('\nTesting complete\n')


    
if __name__ == '__main__':
    main()
