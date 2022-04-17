import numpy as np
import sys

from gaussquad2d import gaussquad1d, gaussquad2d, gaussquad3d
from masternodes import masternodes
from shap import *
sys.path.insert(0, '../util')
sys.path.insert(0, '../mesh')

def mkmaster(mesh, ndim, pgauss=None):

    if ndim == 2:
        if pgauss == None:
            pgauss = mesh['porder']*2

        # Instantiates master data structure and copies the polynomial order and local DG points over to the master structure
        master = {}
        master['porder'] = mesh['porder']
        master['plocvol'], master['tlocvol'], master['plocface'], master['tlocface'], master['corner'], master['perm'], _ = masternodes(master['porder'], ndim)
        master['plocvol'] = master['plocvol'][:,:-1]

        master['gptsface'], master['gwface'] = gaussquad1d(pgauss)
        master['gptsvol'], master['gwvol'] = gaussquad2d(pgauss)

        master['shapface'] = shape1d(master['porder'], master['plocface'], master['gptsface'])
        master['shapvol'] = shape2d(master['porder'], master['plocvol'], master['gptsvol'])
        master['shapvol_nodal'] = shape2d(master['porder'], master['plocvol'], master['plocvol'])   # Useful for computing normal vectors and gradients

        # Generate mass matrices - note the order of the transpose differs from the matlab script because of the C vs Fortran ordering
        master['massface'] = master['shapface'][:,:,0].T@np.diag(master['gwface'])@master['shapface'][:,:,0]
        master['massvol'] = master['shapvol'][:,:,0].T@np.diag(master['gwvol'])@master['shapvol'][:,:,0]
        
        # Convection matrices
        convx = np.squeeze(master['shapvol'][:,:,0]).T@np.diag(master['gwvol'])@np.squeeze(master['shapvol'][:,:,1])
        convy = np.squeeze(master['shapvol'][:,:,0]).T@np.diag(master['gwvol'])@np.squeeze(master['shapvol'][:,:,2])
        master['conv'] = np.concatenate((convx[None, :, :], convy[None, :, :]), axis=0)   # Adding a 3rd empty axis so that they may be concatenated safely

    if ndim == 3:
        if pgauss == None:
            pgauss = mesh['porder']*2

        # Instantiates master data structure and copies the polynomial order and local DG points over to the master structure
        master = {}
        master['porder'] = mesh['porder']
        master['plocvol'], master['tlocvol'], master['plocface'], master['tlocface'], master['corner'], _, master['perm'] = masternodes(master['porder'], ndim)

        master['gptsface'], master['gwface'] = gaussquad2d(pgauss)
        master['gptsvol'], master['gwvol'] = gaussquad3d(pgauss)

        master['shapface'] = shape2d(master['porder'], master['plocface'], master['gptsface'])
        master['shapvol'] = shape3d(master['porder'], master['plocvol'], master['gptsvol'])
        master['shapvol_nodal'] = shape3d(master['porder'], master['plocvol'], master['plocvol'])   # Shape functions evaluated at the nodes for calculating the gradient
        master['phi_inv'] = np.linalg.pinv(master['shapvol'][:, :, 0])

        # Generate mass matrices - note the order of the transpose differs from the matlab script because of the C vs Fortran ordering
        master['massface'] = master['shapface'][:,:,0].T@np.diag(master['gwface'])@master['shapface'][:,:,0]
        master['massvol'] = master['shapvol'][:,:,0].T@np.diag(master['gwvol'])@master['shapvol'][:,:,0]

        # Convection matrices
        convx = np.squeeze(master['shapvol'][:,:,0]).T@np.diag(master['gwvol'])@np.squeeze(master['shapvol'][:,:,1])
        convy = np.squeeze(master['shapvol'][:,:,0]).T@np.diag(master['gwvol'])@np.squeeze(master['shapvol'][:,:,2])
        convz = np.squeeze(master['shapvol'][:,:,0]).T@np.diag(master['gwvol'])@np.squeeze(master['shapvol'][:,:,3])
        master['conv'] = np.concatenate((convx[None, :, :], convy[None, :, :], convz[None, :, :]), axis=0)   # Adding a 3rd empty axis so that they may be concatenated safely

    return master


if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=np.inf, precision=4)
    porder = 3
    mesh = mkmesh_square(porder)
    master = mkmaster(mesh, 2*porder)

    sys.path.insert(0, '../util')
    from import_util import load_mat

    ma1d = load_mat('ma1d')
    mass = load_mat('mass')
    conv = load_mat('conv').transpose((1, 0, 2)).ravel(order='F')

    print(np.allclose(ma1d, master['ma1d']))
    print(np.allclose(mass, master['mass']))
    print(np.allclose(conv, np.ravel(master['conv'])))