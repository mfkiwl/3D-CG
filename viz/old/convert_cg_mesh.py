import numpy as np

def convert_cg_mesh(mesh, field=None):

    mesh['tcg']

    ntlocal = mesh['tlocal'].shape[0]
    tcg_connected = np.zeros((mesh['t'].shape[0]*mesh['tlocal'].shape[0], mesh['t'].shape[1]))

    if field is not None:
        field_out = np.zeros((mesh['pcg'].shape[0], 1))

    for elnum, elem in enumerate(mesh['tcg']):
        for ilocal, localel in enumerate(mesh['tlocal']):
            tcg_connected[elnum*ntlocal+ilocal, :] = elem[localel]
            if field is not None:
                field_out[elem, 0] = field[:, elnum]

    mesh['tcg_connected'] = tcg_connected
    
    if field is not None:
        return mesh, field_out
    else:
        return mesh