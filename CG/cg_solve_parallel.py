import numpy as np
from sympy import interpolate
from elemmat_cg import elemmat_cg
import scipy.sparse
from scipy.sparse import lil_matrix, csr_matrix
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
# import multiprocessing
from numba import njit, prange

# @profile
def assign_dirichlet_bc(master, mesh, A, F, val):
    # A = A.tocsr()
    print('Assigning boundary conditions')
    bdry_faces_start_idx = np.where(mesh['f'][:, -1] < 0)[0][0]

    # Single out only nonzero entries in sparse matrix for dirichlet BC
    nnodes_per_face = mesh['gmsh_mapping'][mesh['elemtype']]['nnodes'][mesh['ndim']-1]
    bdry_faces = mesh['f'][mesh['f'][:, -1] < 0, :]
    for bdry_face_num, face in enumerate(bdry_faces):
        facenum = bdry_face_num + bdry_faces_start_idx + 1      # t2f uses 1-indexing for the faces
        bdry_elem = face[nnodes_per_face]

        # Collect nodes of faces on boundary ONLY
        # Find the index that the face is in in t2f
        loc_face_idx = np.where(mesh['t2f'][bdry_elem, :] == facenum)[0][0]

        # Pull the local nodes on that face from permface - we don't want to include all the nodes in the element
        loc_face_nodes = master['perm'][:, loc_face_idx]

        # Use the perm indices to grab the face nodes from tcg
        face_nodes = mesh['tcg'][bdry_elem][loc_face_nodes]

        # Key takeaway: numpy advanced indexing takes an extremely long time on sparse matrices!!
        for face_node in face_nodes:        # I wish there were a way to vectorize this! Note that there is a feature proposal for a clear() function for csr matrices in the works: https://github.com/scipy/scipy/issues/13746
            A.data[face_node] = [0]*len(A.data[face_node])
            A[face_node, face_node] = 1

        # For example: this took over 11 times longer:
        # A[face_nodes, :] = 0
        # A[face_nodes, face_nodes] = 1

        # And this took almost 4x longer:
        # for face_node in face_nodes:        # I wish there were a way to vectorize this! Note that there is a feature proposal for a clear() function for csr matrices in the works: https://github.com/scipy/scipy/issues/13746
        #     A.data[face_node] = [0]*len(A.data[face_node])
        # A[face_nodes, face_nodes] = 1
        # Note how usually this would be a lot faster because it uses advanced indexing!

        # A.data[face_nodes] = 0
        F[face_nodes, 0] = val

    return A, F

def vissparse(A):
    plt.spy(A)
    plt.show()

# def populate_A_F(ae, fe, A, F)


@njit(parallel=True)
def cg_solve(master, mesh, forcing, param, ndim, solver='dense'):

    if mesh['porder'] == 0:
        raise ValueError('porder > 0 required for continuous galerkin')

    nplocal = mesh['plocal'].shape[0]
    nelem = mesh['t'].shape[0]

    ae = np.zeros((nelem, nplocal, nplocal))
    fe = np.zeros((nplocal, nelem))

    nnodes = mesh['pcg'].shape[0]

    if solver == 'direct':
        raise Exception('Deprecated!')
        A = np.zeros((nnodes, nnodes))
        F = np.zeros((nnodes, 1))

        print('Loading matrix...')
        for i, elem in enumerate(mesh['tcg']):
            A[elem[:, None], elem] += ae[i, :, :]
            F[elem, 0] += fe[:, i]

        print('Assigning boundary conditions...')
        A, F = assign_dirichlet_bc(master, mesh, A, F, 0)

        print('Solving...')
        u = np.linalg.solve(A, F)
    elif solver == 'sparse':

        import time
        start = time.time()

        # Keep it as a lil format until ready to input to conjugate gradient
        # If you have to save the matrix to disk, convert to csr and then back to lil when reimporting
        A= lil_matrix((nnodes, nnodes))
        F = np.zeros((nnodes, 1))

        print('Populating elemental matrices...')
        # for i, elem in enumerate(mesh['tcg']):
        for i in prange(mesh['tcg'].shape[0]):
            elem = mesh['tcg'][i,:]
            if i%1000 == 0:
                print('elem', i, '/', nelem)
            ae, fe = elemmat_cg(mesh['dgnodes'][i,: :], master, forcing, param, ndim)    # Could have used mesh['pcg'][mesh['tcg'][i,:], :] here for the ndoes but more complicated
            A[elem[:, None], elem] += ae
            F[elem, 0] += fe


        # print('Loading matrix...')
        # for i, elem in enumerate(mesh['tcg']):
        #     A[elem[:, None], elem] += ae[i, :, :]
        #     F[elem, 0] += fe[:, i]

        print(time.time()-start)
        A, F = assign_dirichlet_bc(master, mesh, A, F, 0)
        lin = splinalg.aslinearoperator(A)

        # Solve with CG - A entered as a linear operator, but b needs to be dense
        print('Solving...')
        res = splinalg.cg(lin, F, tol=1e-10)       # Play with this tolerance if it is taking too long to run
        if not res[1]:    # Successful exit
            u = res[0][:,None]
            print('Successfully solved with CG...')
        else:
            raise Exception('CG did not converge, reached ' +str(res[1]) + ' iterations')
    
    # Reshape into DG high order data structure
    uh = np.zeros((nplocal, nelem))
    for i in np.arange(nelem):          # Set dirichlet BC
        uh[:, i] = u[mesh['tcg'][i, :], 0]

    print('Done!')

    return uh