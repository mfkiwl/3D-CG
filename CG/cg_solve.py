import numpy as np
import elemmat_cg
from elemforcing_cg import elemforcing_cg
from scipy.sparse import lil_matrix, save_npz, load_npz
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import time
import logging
import pyamg

logger = logging.getLogger(__name__)

def assign_bcs(master, mesh, A, F, approx_sol, issparse=True):
    logger.info('Assigning boundary conditions')
    bdry_faces_start_idx = np.where(mesh['f'][:, -1] < 0)[0][0]

    # Single out only nonzero entries in sparse matrix for dirichlet BC
    nnodes_per_face = mesh['gmsh_mapping'][mesh['elemtype']]['nnodes'][mesh['ndim']-1]
    bdry_faces = mesh['f'][mesh['f'][:, -1] < 0, :]
    for bdry_face_num, face in enumerate(bdry_faces):
        facenum = bdry_face_num + bdry_faces_start_idx + 1      # t2f uses 1-indexing for the faces
        bdry_elem = face[nnodes_per_face]

        # Collect nodes of faces on boundary ONLY
        # Find the index that the face is in in t2f
        loc_face_idx = np.where(mesh['t2f'][bdry_elem, :] == facenum)[0][0]     #try np.abs here?

        # logger.info(master['perm'])     # It might be an issue with perm - it fails at high orders. Look more into what perm does
        # logger.info(loc_face_idx)
        # Pull the local nodes on that face from permface - we don't want to include all the nodes in the element
        loc_face_nodes = master['perm'][:, loc_face_idx]

        # Use the perm indices to grab the face nodes from tcg
        face_nodes = mesh['tcg'][bdry_elem][loc_face_nodes]        
        
        physical_group = -face[-1]
        if physical_group in mesh['dbc'].keys():        # Dirichlet BC

            # Key takeaway: numpy advanced indexing takes an extremely long time on sparse matrices!!
            if issparse:
                for face_node in face_nodes:        # I wish there were a way to vectorize this! Note that there is a feature proposal for a clear() function for csr matrices in the works: https://github.com/scipy/scipy/issues/13746
                    A.data[face_node] = [0]*len(A.data[face_node])
                    A[face_node, face_node] = 1 # Bringing this line out of the for loop slows down by 4x
            else:
                # For example: this took over 11 times longer (but with the sparse matrices instead):
                A[face_nodes, :] = 0
                A[face_nodes, face_nodes] = 1

            F[face_nodes, 0] = mesh['dbc'][physical_group]
            if approx_sol is not None:
                approx_sol[face_nodes] = mesh['dbc'][physical_group]

        elif physical_group in mesh['nbc'].keys():        # Neumann BC
            pts_on_face = mesh['dgnodes'][bdry_elem, loc_face_nodes, :]
            Ff = elemforcing_cg(pts_on_face, master, mesh['nbc'][physical_group], 3)
            F[face_nodes, 0] += Ff
        else:
            raise ValueError('Unknown physical group')

    return A, F, approx_sol

def vissparse(A):
    plt.spy(A)
    plt.show()

count = 0
def call_iter(x):
    global count
    if count %100 == 0:
        logger.info('Iteration' + str(count))
    count += 1

def cg_solve(master, mesh, forcing, param, ndim, outdir, approx_sol=None, buildAF=True, solver='amg', case_name=None):

    if mesh['porder'] == 0:
        raise ValueError('porder > 0 required for continuous galerkin')

    nnodes = mesh['pcg'].shape[0]

    ########## BUILD A AND F ##########

    if buildAF:
        # Keep it as a lil format until ready to input to conjugate gradient
        # If you have to save the matrix to disk, convert to csr and then back to lil when reimporting

        nplocal = mesh['plocal'].shape[0]
        nelem = mesh['t'].shape[0]

        ae = np.zeros((nelem, nplocal, nplocal))
        fe = np.zeros((nplocal, nelem))

        logger.info('Populating elemental matrices...')
        pool = Pool(mp.cpu_count())
        result = pool.map(partial(elemmat_cg.elemmat_cg, mesh['dgnodes'], master, forcing, param, ndim), np.arange(nelem))
        # result = np.asarray(list(map(partial(elemmat_cg.elemmat_cg, mesh['dgnodes'], master, forcing, param, ndim), np.arange(nelem))))
        ae_fe = np.asarray(result)
        ae = ae_fe[:, :, :-1]
        fe = ae_fe[:, :, -1].T

        A= lil_matrix((nnodes, nnodes))
        F = np.zeros((nnodes, 1))

        logger.info('Loading matrix...')
        for i, elem in enumerate(mesh['tcg']):
            if i %10000 == 0:
                logger.info(str(i)+'/'+str(ae.shape[0]))
            A[elem[:, None], elem] += ae[i, :, :]
            F[elem, 0] += fe[:, i]

        logger.info('Saving F...')
        with open(outdir+ case_name + '_F_preBCs.npy', 'wb') as file:
            np.save(file, F)

        logger.info('Saving A...')
        save_npz(outdir + case_name + '_A_preBCs.npz', A.asformat('csr'))
        logger.info('Saved A and F to disk, pre BCs')

    else:
        # Read from disk
        logger.info('Reading A and F from disk')
        with open(outdir + case_name + '_F_preBCs.npy', 'rb') as file:
            F = np.load(file)
        A = load_npz(outdir + case_name + '_A_preBCs.npz').asformat('lil')


    ########## ASSIGN BCs ##########
    A, F, approx_sol = assign_bcs(master, mesh, A, F, approx_sol, issparse=True)

    ########## SOLVE ##########
    logger.info('Solving...')

    start = time.perf_counter()
    if solver=='cg':
        P = lil_matrix((nnodes, nnodes))
        P.setdiag(1/A.diagonal())   # Diagonal preconditioner
        res = splinalg.cg(A, F, M=P, x0=approx_sol, callback=call_iter)
    # else:
    #     lin = splinalg.aslinearoperator(A)
    #     res = splinalg.cgs(lin, F, tol=1e-8)       # Play with this tolerance if it is taking too long to run
        logger.info('Solution time: '+ str(round(time.perf_counter()-start, 5)) +'s')

        if not res[1]:    # Successful exit
            uh = res[0][:,None]
            logger.info('Successfully solved with CG...')
        else:
            raise Exception('CG did not converge, reached ' +str(res[1]) + ' iterations')
    elif solver=='amg':
        ml = pyamg.ruge_stuben_solver(A.tocsr())                    # construct the multigrid hierarchy
        uh = ml.solve(F, tol=1e-8)                          # solve Ax=b to a tolerance of 1e-8
        logger.info('Solution time: '+ str(round(time.perf_counter()-start, 5)) +'s')
    elif solver == 'direct':
        uh = np.squeeze(np.linalg.solve(A.todense(), F))

    return uh, approx_sol
