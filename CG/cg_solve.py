import numpy as np
import elemmat_cg
import quadrature
from scipy.sparse import lil_matrix, save_npz, load_npz
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import time
import logging
import solvers
import gc

logger = logging.getLogger(__name__)
iter_count = 0
last_x = 0

def assign_bcs(master, mesh, A, F, issparse=True):
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

        elif physical_group in mesh['nbc'].keys():        # Neumann BC
            pts_on_face = mesh['dgnodes'][bdry_elem, loc_face_nodes, :]
            Ff = quadrature.elem_surface_integral(pts_on_face, master, np.ones((pts_on_face.shape[0]))*mesh['nbc'][physical_group], 3, returnType='vector')
            F[face_nodes, 0] += Ff
        else:
            raise ValueError('Unknown physical group')

    return A, F

def vissparse(A):
    plt.spy(A)
    plt.show()

def cg_solve(master, mesh, forcing, param, ndim, outdir, casename, buildAF=True, solver='amg', solver_tol=1e-7):
    global last_x
    last_x = 0
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
        start = time.perf_counter()
        for i, elem in enumerate(mesh['tcg']):
            if i %10000 == 0:
                logger.info(str(i)+'/'+str(ae.shape[0]))
            A[elem[:, None], elem] += ae[i, :, :]
            F[elem, 0] += fe[:, i]

        logger.info('Loading matrix took '+ str(time.perf_counter()-start)+' s')

        ########## ASSIGN BCs ##########
        A, F = assign_bcs(master, mesh, A, F, issparse=True)

        logger.info('Saving F post BCs...')
        with open(outdir+ 'F_postBCs_'+casename+'.npy', 'wb') as file:
            np.save(file, F)

        logger.info('Saving A post BCs...')
        save_npz(outdir + 'A_postBCs_'+casename+'.npz', A.asformat('csr'))
        
        logger.info('Done saving A, F')

    else:
        # Read from disk
        logger.info('Reading A and F from disk')
        with open(outdir + 'F_postBCs_'+casename+'.npy', 'rb') as file:
            F = np.load(file)
        A = load_npz(outdir + 'A_postBCs_'+casename+'.npz').asformat('lil')

    ########## SOLVE ##########
    logger.info('Solving with ' + solver)

    logger.info('Prior to solving: Residual norm is {:.5E}'.format(np.linalg.norm(np.linalg.norm(F))))

    sol = solvers.solve(A, F, solver_tol, solver)

    del(A)
    del(F)
    gc.collect()

    return sol
