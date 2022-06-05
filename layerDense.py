import numpy as np
from activationFuncions import Functions as F
# np.random.seed(0)

# X = [[1, 2, 3, 2.5],
#      [2.0, 5.0, -1.0, 2.0],
#      [-1.5, 2.7, 3.3, -0.8]]

# X=np.array(X)

class LayerDense:
    def __init__(self, n_inputs, n_neurones, activitionFunction): 
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurones) #MxN where M is the # of featues in a single training input and N is the # of desired neuorons
        self.biases = np.zeros((1, n_neurones))
        self.activitionFunction = activitionFunction

    def ForwordProp(self, inputs): #in this neural network there will be Mx784 input matrix where M is the # of training inputs
        self.dotOutput = np.dot(inputs, self.weights) + self.biases
        # print(dotOutput.view())
        # print("$"*10)
        self.forwordPropOutput = self.activitionFunction(self.dotOutput)

    # def Cost(self, y):
    #     m = y.shape[1]
    #     cost = -(1/m)*np.sum(y*np.log(a2))
    #     return const
        

    def BackwordProp(self):
        pass

# L = LayerDense(4,10, F.ActivationRelu)
# L.ForwordProp(X)
# print(L.forwordPropOutput.view())