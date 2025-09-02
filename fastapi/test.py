import os
import numpy as np
import matplotlib.pyplot as plt
from config import *

result = np.load("./data/pred_output.npy")

im = plt.imshow(result)
plt.colorbar(im)
plt.show()