from turtle import shape
import numpy as np
from layerDense import LayerDense
import data
from activationFuncions import Functions as F

np.random.seed(0)

class NeuralNetwork:
    def __init__(self, inputs, Tinputs, layers_info):
        self.layers = [LayerDense(n_inputs, n_output, activisionFunc) for n_inputs, n_output, activisionFunc in layers_info]
        self.inputs = inputs
        self.OneHot(Tinputs.reshape((1,-1)))
        self.cost = []

    def OneHot(self, Tinputs):
        shape = (Tinputs.size, 10)
        self.oneHot = np.zeros(shape)
        rows = np.arange(Tinputs.size)
        self.oneHot[rows, Tinputs] = 1


    def Cost(self, y_hat):
        result = -(1/y_hat.shape[0]) * np.sum(np.log(y_hat) * self.oneHot)
        self.cost.append(result)

    def ForwordProp(self):
        self.layers[0].ForwordProp(self.inputs)
        layer1_output = self.layers[0].forwordPropOutput
        self.layers[1].ForwordProp(layer1_output)
        layer2_output = self.layers[1].forwordPropOutput
        self.Cost(layer2_output)

    def ReLU_deriv(self, Z):
        return np.array(Z > 0, dtype = np.float32)

    def BackwordProp(self):
        w2 = self.layers[1].weights
        a1 = self.layers[0].dotOutput
        a2 = self.layers[1].dotOutput
        m = self.inputs.shape[0]
        dz2 = (a2 - self.oneHot)
        self.dw2 = (1/m)*np.dot(dz2.T, a1)
        self.db2 = (1/m)*np.sum(dz2.T, axis = 1, keepdims = True).T
        dz1 = (1/m)*np.dot(w2, dz2.T)*self.ReLU_deriv(a1).T
        self.dw1 = (1/m)*np.dot(self.inputs.T, dz1.T)
        self.db1 = (1/m)*np.sum(dz1, axis = 1, keepdims = True).T


    def UpdatingParms(self, alpha):
        self.layers[0].weights -= alpha * self.dw1
        self.layers[0].biases -=  alpha * self.db1    
        self.layers[1].weights -= alpha * self.dw2 
        self.layers[1].biases -=  alpha * self.db2

    
    def accuracy(self):
        a_out = self.layers[1].forwordPropOutput
        a_out = np.argmax(a_out, 0)  # 0 represents row wise 
        labels = np.argmax(self.oneHot, 0)
        acc = np.mean(a_out == labels)*100
        return acc
    
    def GradientDecent(self, N):
        for i in range(N):
            self.ForwordProp()
            self.BackwordProp()
            self.UpdatingParms(0.002)
            if i%10 == 0:
            # if i == 50:
                # print(self.layers[0].weights[:5])
                # print("*"*50)
                # print(self.layers[0].biases[:5])
                print("cost: ", self.cost[i])
                print("accuracy: ",  self.accuracy())
                # print("-"*50)


if __name__ == "__main__":
    images = data.ImageData("sources/t10k-images.idx3-ubyte")
    imageTitles = data.ImageLabels("sources/t10k-labels.idx1-ubyte")
    # images = np.array([images[0], images[1], images[2]])
    # imageTitles = np.array([imageTitles[0],imageTitles[1], imageTitles[2]]).reshape((-1,1))
    info = [[784,10, F.Relu], [10,10, F.SoftMax]]
    nn = NeuralNetwork(images, imageTitles, info)
    nn.GradientDecent(1000)
