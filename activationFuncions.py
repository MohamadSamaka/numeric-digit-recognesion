import numpy as np

class Functions:
    def Relu(inputs):
        return np.maximum(0, inputs)

    def SoftMax(inputs):
        # print(inputs.view())
        # print("^^^^^^^^"*10)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # probabilities = np.exp(inputs )/ np.sum(inputs)
        return probabilities