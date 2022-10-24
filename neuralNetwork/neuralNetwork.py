import numpy as np
from os import system, name, path, remove
from . import defaults as df, layerDense


"""
    the NerualNetwork class on learning will let the user decide if to use the default values
    that are stored in the default file or it will take the sent alpha and/or # of iterations
    and use it in learning. while in testing it will either use the default modle that has been
    obtained by previous learning process or the user will enter a name of a specefic model.
"""


np.random.seed(0)

def clear():
    if name == 'nt': system('cls')
    else: system('clear')


class NeuralNetwork(df.F.Functions):
    def __init__(self, inputImages = None, inputLabels = None, NetworkInfo = df.Defaults.defaultModel, alpha = None, n_iter = None):    
        self.layers_info = NetworkInfo["layers"]
        self.layers = [layerDense.LayerDense(*layer) for layer in self.layers_info] # layer is a list we use * to unpack it
        self.inputImages = inputImages
        self.alpha =  alpha if alpha else NetworkInfo["alpha"]
        self.n_iter = n_iter if n_iter else NetworkInfo["iterations"]
        if inputLabels is not None: self.OneHotEncodeLabels(inputLabels.reshape((1,-1)))


    def OneHotEncodeLabels(self, inputLabels):
        shape = (inputLabels.size, 10)
        self.oneHot = np.zeros(shape)
        rows = np.arange(inputLabels.size)
        self.oneHot[rows, inputLabels] = 1


    def Cost(self, y_hat):
        y_hat = np.clip(y_hat, 1e-7, 1-1e-7) # if we have a 0 we will get infinity so we clip the vlaues to avoid it
        result = -(1/y_hat.shape[0]) * np.sum(np.log(y_hat) * self.oneHot)
        self.currentCost = result


    def ForwordProp(self):
        self.layers[0].ForwordProp(self.inputImages)
        layer1_output = self.layers[0].forwordPropOutput
        self.layers[1].ForwordProp(layer1_output)
        self.layers[1].forwordPropOutput


    def ReluDerevative(self, Z): #basiclly for val in Z if val > 0 it changes it to 1 if its less change it to 0
        return np.array(Z > 0, dtype = np.float32)


    def BackwordProp(self):
        w2 = self.layers[1].weights
        a1 = self.layers[0].forwordPropOutput
        a2 = self.layers[1].forwordPropOutput
        m = self.inputImages.shape[0]
        dz2 = (a2 - self.oneHot).T
        self.dw2 = (1/m)*np.dot(dz2, a1).T
        self.db2 = (1/m)*np.sum(dz2, axis = 1, keepdims = True).T
        dz1 = np.dot(w2, dz2)*self.ReluDerevative(a1).T
        self.dw1 = (1/m)*np.dot(dz1, self.inputImages).T
        self.db1 = (1/m)*np.sum(dz1, axis = 1, keepdims = True).T


    def UpdatingParms(self, alpha):
        self.layers[0].weights -= alpha * self.dw1
        self.layers[0].biases -=  alpha * self.db1    
        self.layers[1].weights -= alpha * self.dw2 
        self.layers[1].biases -=  alpha * self.db2

    
    def Accuracy(self):
        a_out = self.layers[1].forwordPropOutput
        a_out = np.argmax(a_out, 1) 
        labels = np.argmax(self.oneHot, 1)
        acc = np.mean(a_out == labels)*100
        return acc
                       

    def GradientDecent(self):
        for i in range(self.n_iter):
            self.ForwordProp()
            self.Cost(self.layers[1].forwordPropOutput)
            self.BackwordProp()
            self.UpdatingParms(self.alpha)
            if i%10 == 0:
                self.LearningProcessReport(i + 1)


    def runTest(self, testImages = None, testLabels = None, verbose = True):
        if testImages is not None: self.inputImages = testImages
        if testLabels is not None: self.OneHotEncodeLabels(testLabels.reshape((1,-1)))
        self.ForwordProp()
        if verbose: self.TestReport()
        return np.around(self.layers[1].forwordPropOutput.flatten()*100, decimals = 3).tolist()


    def LearningProcessReport(self, it):
            clear()
            print(f"iteration: {it}\ncost: {self.currentCost}\naccuracy: {self.Accuracy()}")
        

    def TestReport(self):
        print(f"{'*'*40}\ntesting accuracy: {self.Accuracy()}")
        

    def writeModle(self, fName = df.Defaults.defaultModelName): #writing the model after learning is done
        if path.exists(fName): remove(fName)
        np.savez(fName,
        layers = np.asarray([
            [*self.layers_info[0][:3], *self.layers[0].getWeightsBiases()],
            [*self.layers_info[1][:3], *self.layers[1].getWeightsBiases()]
        ], dtype=object),
        iterations = self.n_iter,
        alpha = self.alpha,
        )