import numpy as np
from scipy.sparse import lil_matrix, save_npz, load_npz
import scipy.sparse.linalg as splinalg

A = lil_matrix((10,10))

# Set up A matrix - simple second order linear operator. A has to be symmetric for conjugate gradient to work
ones = np.ones(A.shape[0])
A.setdiag(-2*ones, k=0)
A.setdiag(ones, k=-1)
A.setdiag(ones, k=1)

# Assign an arbitrary RHS
b = np.arange(A.shape[0])[:,None]

A[0,:] = 0
A[-1,:] = 0

# Dirichlet BCs
A[0,0] = 1
b[0] = 1

A[-1,-1] = 1
b[0] = 0

# Direct solution
A_dense = A.todense()
sol_direct = np.linalg.solve(A_dense, b)

# Iterative solution using conjugate gradient
print(splinalg.cg(A, b)[1])
sol_sparse = splinalg.cg(A, b)[0][:,None]
print(sol_sparse)
print()
print(sol_direct)

print('Direct solution:', np.linalg.norm(A_dense@sol_direct-b))
print('Iterative solution:', np.linalg.norm(A_dense@sol_sparse-b))
