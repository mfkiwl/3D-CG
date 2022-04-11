import numpy as np
from scipy.sparse import lil_matrix, save_npz, load_npz
print('importing')
import pickle
print('imported')

outdir = 'out/'
print('Reading A and F from disk')
with open(outdir + 'F_preBCs.npy', 'rb') as file:
    F = np.load(file)
A = load_npz(outdir + 'A_preBCs.npz')
with open(outdir+'sol', 'rb') as file:
    x = pickle.load(file)

print('calculating residual')
print(A.dtype)
print(x.shape)
residual = A@x[:,None] - F
print('norm')
res_norm = np.linalg.norm(residual)

print(res_norm)