# Finding the sim root directory
import sys
from pathlib import Path
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('util')))
sys.path.append(str(sim_root_dir.joinpath('mesh')))
sys.path.append(str(sim_root_dir.joinpath('master')))
sys.path.append(str(sim_root_dir.joinpath('viz')))
sys.path.append(str(sim_root_dir.joinpath('CG')))
import masternodes
import cgmesh
import helper
import numpy as np
import logging
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp
import quadrature

logger = logging.getLogger(__name__)

def compute_normal_derivatives(mesh_vol, master_vol, mesh_face, vector_field, faces, element_normal_dot_surface_normal=1):   # Accounts for the outward facing normal being the surface inward facing normal - will need to flip on the plane.
    # Get normal vectors for the HO pts on each face using compute_normal_derivatives()

    idx_set = []
    nnodes_per_face = mesh_vol['gmsh_mapping'][mesh_vol['elemtype']]['nnodes'][mesh_vol['ndim']-1]

    for iface, face in enumerate(faces):
        bdry_elem = face[nnodes_per_face+1]

        # Collect nodes of faces on boundary ONLY
        # Find the index that the face is in in t2f
        loc_face_idx = np.where(mesh_vol['t2f'][bdry_elem, :] == face[0])[0][0] # Remember that the first element in the face is the global index, as built in the fist line of the function
        idx_set.append((bdry_elem, loc_face_idx, iface))

    logger.info('Computing normal vectors')
    pool = Pool(mp.cpu_count())
    result = pool.map(partial(quadrature.get_elem_face_normals, mesh_vol['dgnodes'], master_vol, mesh_vol['ndim']), idx_set)
    # result = map(partial(quadrature.get_elem_face_normals, mesh_vol['dgnodes'], master_vol, mesh_vol['ndim']), idx_set)
    normals_array = np.hstack(result)

    # Reassemble into the pcg format
    normals_pcg = np.squeeze(helper.reshape_field(mesh_face, normals_array[None,:,:], 'to_column', 'vectors', porder=None, dim_override=3))

    # Take dot(grad, normal) to find the normal derivative
    normal_derivative_qty = np.sum(vector_field*normals_pcg, axis=1)[:,None]

    return normal_derivative_qty*element_normal_dot_surface_normal

def extract_surfaces(mesh, master, face_groups, case, field, return_normal_quantity=False, element_normal_dot_surface_normal=1):   # Accounts for the outward facing normal being the surface inward facing normal - will need to flip on the plane.):
    """
    face_groups can either be a list of individual faces to plot, or a list of physical groups to plot

    'case' can either be 'pg', meaning that the items in 'face_list' represent indices into the physical group, or 'face', meaning that
    each item in 'face_list' represents a single face index to visualize. The latter allows finer grain control over what is visualized.
    
    'field' is the full scalar field of size mesh['pcg'].shape[0]
    """

    f_with_index = np.concatenate((np.arange(mesh['f'].shape[0])[:,None]+1, mesh['f']), axis=1)

    # Extract those faces from mesh.f. Note that we cut off the physical group index as this is no longer necessary information
    if case == 'face':
        logger.info('Extracting surfaces based on face: did you make sure to 0-index faces?')
        faces = f_with_index[face_groups, :-1]   # Specific faces input to visualize    face_groups HAVE TO BE 0-INDEXED!!
    elif case == 'pg':
        faces = []
        for group in face_groups:
            face_indices = np.where(f_with_index[:,-1] == -group)[0]
            faces.append(f_with_index[face_indices, :-1])   # Specific faces input to visualize

        faces = np.vstack(faces)
   
    # We now have a 2D connectivity matrix. Turn this into a HO CG mesh using a 2D version of cgmesh
    logger.info('Creating high order 2D surface mesh')
    mesh_face, face_field = cgmesh.cgmesh(mesh, faces, master, case='surface_mesh', field=field)
    mesh_face['porder'] = mesh['porder']
    mesh_face['ndim'] = mesh['ndim'] - 1
    mesh_face['plocal'], mesh_face['tlocal'], _, _, mesh_face['corner'], _, _ = masternodes.masternodes(mesh_face['porder'], mesh_face['ndim'])

    if return_normal_quantity:
        face_field = compute_normal_derivatives(mesh, master, mesh_face, face_field, faces, element_normal_dot_surface_normal)

    return mesh_face, face_field
    