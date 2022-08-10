from scipy.io import savemat
import numpy as np
import glob
import os

filename = 'square'

with open('square' + '.npy', 'rb') as f:
    p = np.load(f)
    t = np.load(f)

d = {'p': p, 't': t}

savemat(filename + '.mat', d)

# npzFiles = glob.glob("*.npy")
# for f in npzFiles:
#     fm = os.path.splitext(f)[0]+'.mat'
#     p = np.load(f)
#     t = np.load(f)
#     savemat(fm, p)
#     savemat(fm, t)
#     print('generated ', fm, 'from', f)
