############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/23/2022
#   file: ann.py
#   Description: main class for artificial neural network
#############################################################

from ann import ANN

def main():
    '''main of the program'''

    training_path = 'data/iris/iris-train.txt'
    testing_path = 'data/iris/iris-test.txt'
    attributes_path = 'data/iris/iris-attr.txt'
    debugging = False


    print('\nLearning the decision tree...\n')
    # run the program
    dtl = Learner(attributes_path, training_path, testing_path, debugging)
    tree = dtl.learn()

    print('\nPrinting the decision tree...\n')
    tree.print_tree()

    print('\nTesting the tree on training data\n')
    # testing tree on training data
    training_acc = dtl.test(tree, dtl.training)
    print('\nTraining accuracy: ', training_acc)

    print('\nTesting the tree on testing data\n')
    # testing tree on test data
    testing_acc = dtl.test(tree, dtl.testing)
    print('\nTesting accuracy: ', testing_acc)

    print('\nPrinting the decision tree rules\n')
    # print the rules
    dtl.tree_to_rules(tree)    
    tree.print_rules()
    
    print('\nPruning the tree...\n')
    # prune the tree
    dtl.rule_post_pruning(tree, dtl.testing)
    tree.rules = consolidate_rules(tree.rules)
    tree.print_rules()

    print('\nTesting the rules on training data\n')
    # testing tree on test data
    training_acc = dtl.test(tree, dtl.training)
    print('\nTraining accuracy: ', training_acc)

    print('\nTesting the rules on testing data\n')
    # testing tree on test data
    testing_acc = dtl.test(tree, dtl.testing)
    print('\nTesting accuracy: ', testing_acc)


    
if __name__ == '__main__':
    main()
