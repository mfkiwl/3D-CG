import shelve

def load_mesh():
    with shelve.open('../mesh/mesh_mat') as shelf:
        mesh_mat = shelf['mesh_mat']
    return mesh_mat