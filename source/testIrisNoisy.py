############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/23/2022
#   file: ann.py
#   Description: main class for artificial neural network
#############################################################

from ann import ANN
from utils import corrupt_data

def main():
    '''main of the program'''

    training_path = 'data/iris/iris-train.txt'
    testing_path = 'data/iris/iris-test.txt'
    attributes_path = 'data/iris/iris-attr.txt'
    debugging = False
    validation = True # higher accuracy when test data is used for vaildation

    # create learner
    dtl = Learner(
        attributes_path, 
        training_path, 
        testing_path, 
        debugging,
        validation
    )

    # start of the experiment 
    for p in range(0, 22, 2):


        print('////////////////////////////////////////////////////////////')
        # corrupt the data'
        print('\nCorrupting the data by changing from the correct class to another class...')
        print('\nPercentage of corruption: {}%'.format(p))
        dtl.training = corrupt_data(dtl.training, dtl.get_classes(), p / 100.)
        # print('\nCorrupted data:')
        # print(dtl.training)

        print('\nLearning the decision tree...\n')
        # run the program
        tree = dtl.learn()

        print('\nTesting the tree on uncorrupted testing data\n')
        # testing tree on test data
        testing_acc = dtl.test(tree, dtl.testing)
        print('\nTesting accuracy: ', testing_acc)

        print('\nPrinting the decision tree rules\n')
        # print the rules
        dtl.tree_to_rules(tree)    
        # tree.print_rules()

        print('\nPruning the tree...\n')
        # prune the tree
        tree = dtl.rule_post_pruning(tree, dtl.validation)
        tree.rules = consolidate_rules(tree.rules)
        # tree.print_rules()

        print('\nTesting the rules on uncorrupted testing data\n')
        # testing tree on test data
        testing_acc = dtl.test(tree, dtl.testing)
        print('\nTesting accuracy: ', testing_acc)

    
if __name__ == '__main__':
    main()
