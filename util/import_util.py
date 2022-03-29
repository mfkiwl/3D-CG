import numpy as np
from scipy.io import loadmat
import shelve
import sys
sys.path.insert(0, '../util')
from gmshcall import gmshcall
import pickle

def process_mesh(filename, ndim, nelem, nface=None):
    """
    ndim: dimensionality of the mesh
    nelem: element ID of the highest dimensional mesh element - 4 for tets, 2 for triangles
    nface: element ID of the highest dimensional mesh element - 2 for triangles, 1 for lines
    """

    p, t, pg = gmshcall(filename + '.msh', ndim, nelem, nface)

    mesh = {'p': p, 't': t, 'pg': pg}

    # with open(filename + '.npy', 'wb') as f:
    #     np.save(f, p)
    #     np.save(f, t)

    with open(filename + '_pg', 'wb') as file:
        pickle.dump(pg, file)

    return mesh

def load_processed_mesh(filename):
    with open(filename+'.npy', 'rb') as f:
        p = np.load(f)
        t = np.load(f)

    # with shelve.open(filename + '_pg', 'r') as shelf:
    #     pg = shelf['pg']

    # mesh = {'p': p.transpose(), 't': t.transpose(), 'physical_groups': pg}
    mesh = {'p': p.transpose(), 't': t.transpose()}

    return mesh


def load_mesh_structure_from_mat():
    matlab_files_to_import = ['mesh_dgnodes', 'mesh_f', 'mesh_t',
                              'mesh_t2f', 'mesh_tlocal', 'mesh_plocal', 'mesh_p']

    mesh_mat = {}
    for mat in matlab_files_to_import:
        mesh_mat[mat.replace('mesh_', '')] = loadmat('../data/'+mat)[mat]

    with shelve.open('../mesh/mesh_mat') as shelf:
        shelf['mesh_mat'] = mesh_mat

def load_mat(fstem, key=None):
    if key:
        return loadmat(fstem+'.mat')[key]
    else:
        return loadmat(fstem+'.mat')[fstem]

def load_mesh():
    with shelve.open('../mesh/mesh_mat') as shelf:
        mesh_mat = shelf['mesh_mat']
    return mesh_mat

if __name__ == '__main__':
    load_mesh_structure_from_mat()

    # with shelve.open('../mesh/mesh_mat') as shelf:
    #     mesh_mat = shelf['mesh_mat']
    #     print(mesh_mat.keys())