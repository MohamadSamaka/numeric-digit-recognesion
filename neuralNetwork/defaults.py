from numpy import load
from . import data
from . import activationFuncions as F
from copy import deepcopy
_parent = __name__.split('.')[0]

class Defaults:
    trainingImages = data.ImageData(f"{_parent}/sources/train-images.idx3-ubyte")
    trainingLabels = data.ImageLabels(f"{_parent}/sources/train-labels.idx1-ubyte")
    testingImages = data.ImageData(f"{_parent}/sources/t10k-images.idx3-ubyte")
    testingLabels = data.ImageLabels(f"{_parent}/sources/t10k-labels.idx1-ubyte")
    defaultModelName = f"{_parent}/models/default.npz"
    defaultNetworkInfo = {"layers": [[784,10, F.Functions.Relu], [10,10, F.Functions.SoftMax]],
                           "alpha": 0.1,
                           "iterations": 500
                           }
    try:
        handle = load(defaultModelName, allow_pickle=True)
        defaultModel = deepcopy({"layers": handle.f.layers, "alpha": handle.f.alpha, "iterations": handle.f.iterations})
        handle.close()
        del handle
    except:
        defaultModel = defaultNetworkInfo