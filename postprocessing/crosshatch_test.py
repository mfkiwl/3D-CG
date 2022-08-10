from numpy.random import rand
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ax = plt.gca()
ax.imshow(rand(50,50),interpolation='bicubic')
for i,j in np.floor(50*rand(10,2)).astype('int'):
    ax.add_patch(mpl.patches.Rectangle((i-.5, j-.5), 1, 1, hatch='///////', fill=False, snap=False))

plt.show()