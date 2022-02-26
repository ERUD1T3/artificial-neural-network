############################################################
#   Dev: Josias Moukpe
#   Class: Machine Learning
#   Date: 2/23/2022
#   file: ann.py
#   Description: main class for artificial neural network
#############################################################

class ANN:
    '''
    Feed Forward Artificial Neural Network Class
    1 Input, 1 Hidden, 1 Output Layer
    '''
    def __init__(
        self, 
        num_input, 
        num_hidden, 
        num_ouptut, 
        lr, 
        momentum, 
        debug=True
    ) -> None:
        
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_ouptut
        self.lr = lr
        self.momentum = momentum
        self.debug = debug
        