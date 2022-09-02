from numpy import load
import data
from activationFuncions import Functions as F

class Defaults:
    trainingImages = data.ImageData("sources/train-images.idx3-ubyte")
    trainingLabels = data.ImageLabels("sources/train-labels.idx1-ubyte")
    testingImages = data.ImageData("sources/t10k-images.idx3-ubyte")
    testingLabels = data.ImageLabels("sources/t10k-labels.idx1-ubyte")
    defaultModelName = "models/default.npz"
    defaultNetworkInfo = {"layers": [[784,10, F.Relu], [10,10, F.SoftMax]],
                           "alpha": 0.1,
                           "iterations": 1500
                           }
    try:
        defaultModel = load(defaultModelName, allow_pickle=True)
    except:
        defaultModel = defaultNetworkInfo