import numpy as np
import idx2numpy
import matplotlib.pyplot as plt

imagefile = 'sources/t10k-images.idx3-ubyte'
imagearray = idx2numpy.convert_from_file(imagefile)

plt.imshow(imagearray[100], cmap=plt.cm.binary)
plt.show()
