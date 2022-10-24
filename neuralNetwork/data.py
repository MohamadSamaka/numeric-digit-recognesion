import numpy as np
import idx2numpy
import os

def ImageData(src):
    imagearray = idx2numpy.convert_from_file(src)
    imagearray = imagearray.astype('float32')/255.0
    imagearray = np.reshape(imagearray, (len(imagearray),-1))
    return imagearray

def ImageLabels(src):
    ImageTitles = idx2numpy.convert_from_file(src).reshape((-1,1))
    return ImageTitles
