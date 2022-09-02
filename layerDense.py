import numpy as np
from activationFuncions import Functions as F

class LayerDense:
    def __init__(self, n_inputs, n_neurones, activitionFunction, weights = None, biases = None):
        if weights is not None and biases is not None:
            self.weights = weights
            self.biases = biases
        else:
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurones) #MxN where M is the # of featues in a single training input and N is the # of desired neuorons
            self.biases = np.zeros((1, n_neurones))
        self.activitionFunction = activitionFunction
        


    def ForwordProp(self, inputs): #in this neural network there will be Mx784 input matrix where M is the # of training inputs
        self.dotOutput = np.dot(inputs, self.weights) + self.biases
        self.forwordPropOutput = self.activitionFunction(self.dotOutput)
