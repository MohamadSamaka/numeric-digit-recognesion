import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

def ImageData(src):
    # imagefile = 'sources/t10k-images.idx3-ubyte'
    imagearray = idx2numpy.convert_from_file(src)
    imagearray = imagearray.astype('float32')/255.0
    imagearray = np.reshape(imagearray, (len(imagearray),-1))
    return imagearray

def ImageLabels(src):
    # lablfile = 'sources/t10k-labels.idx1-ubyte'
    ImageTitles = idx2numpy.convert_from_file(src).reshape((-1,1))
    return ImageTitles


