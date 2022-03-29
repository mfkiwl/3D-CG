import numpy as np
from import_util import load_mat


def cgmesh(mesh):
    """
    Steps:
    - Reshape ph to a ndgnodesx2 array of all the coordinates of the dgnodes, including duplicate points on the faces
    - Round the coordinate values to make it easier for the duplicated nodes to match
    - Computes the unique coordinate array, removing duplicates. The key piece that enables this method is that the np.unique function also retuns the inverse indices, an array of the same legnth as the input, mapping the indices of the sorted, unique elements to their places in the original array (can contain duplicates).
    - Since we have put the dgnodes in order going down the reshaped array, our connectivity matrix at this point is just np.arange(ndgodes).reshape((nelem, nplocal)). The tricky part is to re-map the indices of the old connectivity matrix to the new indices in the unique array.
    - But, since the inverse index array already contains this mapping, we can just reshape it as above and it will be correct.

    Works for 3D as well!
    """

    nplocal = mesh['dgnodes'].shape[1]

    ndim = mesh['dgnodes'].shape[2]
    ph = np.transpose(mesh['dgnodes'], (2, 1, 0))
    ph = np.ravel(ph, order='F').reshape((-1, ndim))    # ndim accounts for 3D as well

    _, unique_idx, inverse_idx = np.unique(np.round(ph, 6), axis=0, return_index=True, return_inverse=True)
    ph_unique = ph[unique_idx,:]

    tcg = np.reshape(inverse_idx, (-1, nplocal))

    mesh['pcg'] = ph_unique
    mesh['tcg'] = tcg

    return mesh

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '../mesh')
    from mkmesh_sqmesh_gmsh import mkmesh_square
    
    sys.path.insert(0, '../util')
    from import_util import load_mat

    np.set_printoptions(suppress=True, precision=10, linewidth=np.inf)

    tcg_mat = load_mat('tcg')
    pcg = load_mat('pcg')

    mesh = mkmesh_square(3)
    tcg = cgmesh(mesh)

    print(np.allclose(mesh['pcg'], pcg, rtol=1e-13, atol=4e-15))        # DIDN'T WORK!!!!!!! TOLERANCE WAS TOO HIGH!!!!
    print(np.allclose(mesh['tcg']+1, tcg_mat))
    