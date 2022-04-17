import numpy as np
import sys
from pathlib import Path
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('util')))
import gmshwrite

def cgmesh(mesh, t_linear=None, master=None, case='volume_mesh', field=None, type=None):
    """
    Steps:
    - Reshape ph to a ndgnodesx2 array of all the coordinates of the dgnodes, including duplicate points on the faces
    - Round the coordinate values to make it easier for the duplicated nodes to match
    - Computes the unique coordinate array, removing duplicates. The key piece that enables this method is that the np.unique function also retuns the inverse indices, an array of the same legnth as the input, mapping the indices of the sorted, unique elements to their places in the original array (can contain duplicates).
    - Since we have put the dgnodes in order going down the reshaped array, our connectivity matrix at this point is just np.arange(ndgodes).reshape((nelem, nplocal)). The tricky part is to re-map the indices of the old connectivity matrix to the new indices in the unique array.
    - But, since the inverse index array already contains this mapping, we can just reshape it as above and it will be correct.

    Works for 3D as well!
    """

    if case == 'volume_mesh':
        nplocal = mesh['dgnodes'].shape[1]

        ndim = mesh['ndim']
        ph = np.transpose(mesh['dgnodes'], (2, 1, 0))
        if type == 'surface_mesh':
            ph = np.ravel(ph, order='F').reshape((-1, mesh['p'].shape[1]))
        else:
            ph = np.ravel(ph, order='F').reshape((-1, ndim))


        _, unique_idx, inverse_idx = np.unique(np.round(ph, 6), axis=0, return_index=True, return_inverse=True)
        ph_unique = ph[unique_idx,:]    # This is what caused for big trouble when I used the output of np.unique directly - the points had been rounded

        tcg = np.reshape(inverse_idx, (-1, nplocal))

        mesh['pcg'] = ph_unique
        mesh['tcg'] = tcg
        return mesh


    elif case == 'surface_mesh':    # In the case of the surface mesh, the node numbers will be a subset of all nodes, and nplocal will be different
        nplocface = master['plocface'].shape[0]
        numel = t_linear.shape[0]
        nnodes_per_face = mesh['nnodes_per_face']
        ndim = mesh['ndim']

        # Note: t_linear has the element that it belongs to tacked onto the last column - this is taken from a list of faces remember

        # ph is an array of all the points in the mesh
        # ph = np.zeros((nplocface*numel, ndim))
        th = np.zeros((nplocface*numel)).astype(np.int32)
        
        face_mesh = {}
        face_mesh['t'] = np.zeros((t_linear.shape[0], nnodes_per_face)).astype(np.int32)

        for iface, face in enumerate(t_linear):
            facenum = face[0]   # Global facenumber, as this got lost when we sliced the face array
            bdry_elem = face[-1]
            face_elem = face[1:-1]
            face_mesh['t'][iface, :] = face_elem    # face_mesh['t] contains the connectivity in the GLOBAL volume node array

            loc_face_idx = np.where(mesh['t2f'][bdry_elem, :] == facenum)[0][0]
            # Pull the local nodes on that face from permface - we don't want to include all the nodes in the element
            loc_face_nodes = master['perm'][:, loc_face_idx]

            # Use the perm indices to grab the face nodes from tcg
            face_nodes = mesh['tcg'][bdry_elem][loc_face_nodes] # in pcg
            
            th[nplocface*iface:nplocface*(iface+1)] = face_nodes

        # Build mesh.p - renumber faces
        p_unique_faces, __, p_inverse_idx = np.unique(face_mesh['t'].ravel(), return_index=True, return_inverse=True)
        face_mesh['p'] = mesh['p'][p_unique_faces,:]
        face_mesh['t'] = np.reshape(p_inverse_idx, (-1, nnodes_per_face))    # Resets the numbering of the faces on the surface mesh

        p_unique_faces, unique_idx, inverse_idx = np.unique(th, return_index=True, return_inverse=True)

        tcg_faces = np.reshape(inverse_idx, (-1, nplocface))    # Resets the numbering of the faces on the surface mesh

        pcg_faces = mesh['pcg'][p_unique_faces]
        field_faces = field[p_unique_faces]     # Works for both scalar and vector arrays

        face_mesh['tcg'] = tcg_faces
        face_mesh['pcg'] = pcg_faces

        return face_mesh, field_faces

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

    print(np.allclose(mesh['pcg'], pcg, rtol=1e-13, atol=4e-15))
    print(np.allclose(mesh['tcg']+1, tcg_mat))
    