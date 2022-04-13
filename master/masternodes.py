import numpy as np
from scipy.misc import face

# TODO: extend to account for elements with curved faces and a different porder on the faces (morder)

def masternodes(porder, ndim):

    if ndim == 2:
        if porder == 0:
            plocal = np.array([1/3, 1/3, 1/3])
            tlocal = np.array([0, 1, 2])
        else:
            pts1d = np.linspace(0, 1, num=porder+1)
            x, y = np.meshgrid(pts1d, pts1d, indexing='ij')     # Produces two arrays with matrix-style indexing, with x, the first index, changing down the row
            plocal = np.asarray((x.ravel(order='F'), y.ravel(order='F'))).T     # Arrays flattened column-major, with the row changing the fastest as if you were traveling across the x-axis and then up in y. This correlates to Matlab/FORTRAN ordering.
            plocal = np.concatenate((1-np.sum(plocal,axis=1)[:,None], plocal), axis=1)  # Adds the first column to the array as the parametric value: 1-x-y
            plocal = plocal[plocal[:,0]>=0, :]  # Takes out all points above diagonal - with x+y > 1

            # Generate tlocal
            tloclist = []
            # Number of pts that you'll traverse in the row depending on which row you're on. Think of this as like strides in a numpy matrix - the stride from one node to the next in a single row is 1, whereas the stride to the next row is DEPENDENT on what row you're on. This is the only difference between it and the numpy stride.

            layer_offset = 0
            for i in np.arange(0, porder+1):
                stride = porder+1-i

                # Add parity 1 elements
                for j in np.arange(0, stride-1):
                    tloclist.append(np.array([j, j+1, j+stride])+layer_offset)

                # Add parity -1 elements
                for j in np.arange(0, stride-2):
                    tloclist.append(np.array([j+1, j+1+stride, j+stride])+layer_offset)
                layer_offset += stride

            tlocal = np.asarray(tloclist)

            plocface = np.linspace(0, 1, num=porder+1, endpoint=True)
            tlocface =np.concatenate((np.arange(porder)[:, None], np.arange(1, porder+1)[:, None]), axis=1)

            # Make permnodes - the ith corner points going CCW has a 1 in the ith column of pl2d.
            permnodes = []
            for col in plocal.T:
                permnodes.append(np.where(np.isclose(col,1))[0][0])
            permnodes = np.asarray(permnodes)

            # Make permedge
            permedge = []
            for i, col in enumerate(plocal.T):
                # For each column in pl2d, we need to pull out the points that have zeros in them. For points in the same column, they will lie on the same face.
                iface_node_idx = np.where(np.isclose(col,0))[0]

                # Next we need to check if the points are being indexed CCW. This relies on the fact that for the column/"axis" that is held constant at zero, the column one to the right (can roll over of course) should be decreasing. For ex: on diagonal of triangle, traversing the face nodes CCW means x is decreasing.
                # So, we sort on the i+1 th column and then reverse it so that it is decreasing.
                # Just the plocal nodes on the ith face
                nodes_on_ith_face = plocal[iface_node_idx, :]
                sorted_idx = np.argsort(
                    np.roll(nodes_on_ith_face, -(i+1), axis=1)[:, 0])
                sorted_idx = sorted_idx[::-1]
                permedge.append(iface_node_idx[sorted_idx])

            permedge = np.asarray(permedge).T
            
            # Since the parametric coordinates have served their purpose, remove them (the first column) of the local point matrices while returning
            plocal = plocal[:,1:]
            plocal = np.concatenate((plocal, np.zeros((plocal.shape[0], 1))), axis=1)

        return plocal, tlocal, plocface, tlocface, permnodes, permedge, None


    elif ndim == 3:
        zs = lambda npl, j: npl-j

        if porder == 0:     # Not really sure what the point of this is
            plocal = np.array([1/3, 1/3, 1/3, 1/3])
            tlocal = np.array([0, 1, 2, 3])
        else:
            pts1d = np.linspace(0, 1, num=porder+1)
            # Produces two arrays with matrix-style indexing, with x, the first index, changing down the row
            x, y, z = np.meshgrid(pts1d, pts1d, pts1d, indexing='ij')
            # Arrays flattened column-major, with the row changing the fastest as if you were traveling across the x-axis and then up in y. This correlates to Matlab/FORTRAN ordering.
            plocal = np.asarray((x.ravel(order='F'), y.ravel(order='F'), z.ravel(order='F'))).T
            # Adds the first column to the array as the parametric value: 1-x-y
            plocal = np.concatenate((1-np.sum(plocal, axis=1)[:, None], plocal), axis=1)
            # Takes out all points above diagonal - with x+y > 1
            # Change to use np.isclose
            eps = 1e-6
            plocal = plocal[plocal[:, 0] >= 0-eps, :]       # Accounts for floating point error

            # Generate tlocal
            tloclist = []
            # Number of pts that you'll traverse in the row depending on which row you're on. Think of this as like strides in a numpy matrix - the stride from one node to the next in a single row is 1, whereas the stride to the next row is DEPENDENT on what row you're on. This is the only difference between it and the numpy stride.

            # Overall what we're doing is basically just iterating through all the points in C style format [z, y, x], with [x] changing fastest in the innermost loop. For each x value, compute all the tetrahedra in front of it in y and z.

            z_layer_offset = 0        # Total number of nodes at all the z levels below
            for k in np.arange(0, porder+1):        # For every row in z
                loc2d_porder = porder-k
                y_layer_offset = 0        # Total number of nodes at all the y levels below the current y-level
                for j in np.arange(porder + 1 - k):      # At each level, effective number of 1D plocal nodes
                    ys = loc2d_porder + 1 -j      # "alias" for y_stride - number of the idx you need to increment to reach the same point on the next y level                    
                    npl = int((loc2d_porder+1)*(loc2d_porder+2)/2)         # "alias" for 'number per layer' - number of the idx you need to increment to reach the same point on the next z level if the next level WERE THE SAME SIZE (this will be modified later by the zs function)
                    
                    offset = z_layer_offset + y_layer_offset
                    # Add parity 1 elements - need 3 here
                    
                    for i in np.arange(ys-1):
                        bxpt1 = i
                        bxpt2 = i+1
                        bxpt3 = i+ys
                        bxpt4 = i+1+ys
                        bxpt5 = i+zs(npl, j)
                        bxpt6 = i+1+zs(npl, j)
                        bxpt7 = i+ys+zs(npl, j+1)
                        bxpt8 = i+1+ys+zs(npl, j+1)

                        # Strategy: divide the master element into sub-cubes and iterate through each one by row. For each row of sub-cubes, there will be two partial cubes at the end - one having 5 tets and the other having only 1. These are represented below
                        # This can be easily tested by making a 8-point mesh with a single cube and then building the connectivity matrix like the indexing here.
                        
                        tloclist.append(np.array([bxpt1, bxpt2, bxpt3, bxpt5])+offset)
                        
                        if i < ys-2:
                            tloclist.append(np.array([bxpt3, bxpt2, bxpt4, bxpt5])+offset)
                            tloclist.append(np.array([bxpt5, bxpt2, bxpt4, bxpt6])+offset)
                            tloclist.append(np.array([bxpt4, bxpt3, bxpt5, bxpt7])+offset)
                            tloclist.append(np.array([bxpt5, bxpt7, bxpt6, bxpt4])+offset)

                            if i < ys-3:
                                tloclist.append(np.array([bxpt6, bxpt7, bxpt8, bxpt4])+offset)

                    y_layer_offset += ys
                z_layer_offset += npl

            tlocal = np.asarray(tloclist)

            plocface, tlocface,_,_,_,_,_ = masternodes(porder, 2)

            permface = []

            p1 = np.array([0, 0, 0])
            p2 = np.array([1, 0, 0])
            p3 = np.array([0, 1, 0])
            p4 = np.array([0, 0, 1])
            p = np.array([p1, p2, p3, p4]).astype(np.float)
            face_vectors = np.array([[p4-p2,p3-p2,np.cross(p4-p2,p3-p2)],
                                    [p4-p3,p1-p3,np.cross(p4-p3,p1-p3)],
                                    [p4-p1,p2-p1,np.cross(p4-p1,p2-p1)],
                                    [p2-p1,p3-p1,np.cross(p2-p1,p3-p1)]])

            p0_vec = np.array([p2, p3, p1, p1])

            permface = np.zeros((plocface.shape[0], 4)).astype(np.int)  # 4 is the number of faces in a tet, no use in using a variable to set it because the code can only handle tets

            for i, col in enumerate(plocal.T):
                iface_node_idx = np.where(np.isclose(col, 0))[0]    # list of points on ith face
                pts_on_face = plocal[iface_node_idx,1:]     # 1-indexing because the first column still has the parametric coord index tacked on

                # Affine transformation matrix - in regular form Ap = r
                A = face_vectors[i,:,:]

                plf_transformed = np.around((pts_on_face -p0_vec[i,:])@np.linalg.inv(A), decimals=6)
                plocface_rounded = np.around(plocface, decimals=6)

                found_idx = np.zeros([plocface_rounded.shape[0]]).astype(np.int)
                # Loop through each point and find which index it corresponds to in the master
                for iloc_pt, loc_pt in enumerate(plocface_rounded):
                    found_idx[iloc_pt] = np.where(np.all(plf_transformed == loc_pt[None,:], axis=1))[0][0]

                # Convert back to the global index - index perm into iface_node_idx
                permface[:,i] = iface_node_idx[found_idx]

            plocal = plocal[:,1:]
            plocal_rounded = np.around(plocal, decimals=6)
   
            permnodes = np.zeros((4)).astype(np.int32)
            for ivertex, _ in enumerate(permnodes):
                # print(permnodes[ivertex])
                permnodes[ivertex] = np.where(np.all(plocal_rounded==p[ivertex,:][None,:], axis=1))[0][0]

            # Since the parametric coordinates have served their purpose, remove them (the first column) of the local point matrices while returning
            plocal = plocal
            plocface = plocface[:,:-1]

    return plocal, tlocal, plocface, tlocface, permnodes, None, permface


if __name__ == '__main__':

    porder = 3
    dim=3
    plocal, tlocal, plocface, tlocface, corner, permedge, perm = masternodes(porder, dim)

    print('porder')
    print(porder)
    print('dim')
    print(dim)
    print('plocal')
    print(plocal)
    print()
    print('tlocal')
    print(tlocal)
    print()
    print('plocface')
    print(plocface)
    print()
    print('tlocface')
    print(tlocface)
    print()
    print('corner')
    print(corner)
    print()
    print('permedge')
    print(permedge)
    print()
    print('perm')
    print(perm)
    print()

    exit()

    f_idx_template = np.array([[1, 3, 2],    # Nodes on face 0
                               [2, 3, 0],      # Nodes on face 1
                               [0, 3, 1],      # Nodes on face 2
                               [0, 1, 2]])     # Nodes on face 3

    # print(f_idx_template+1)


    plocal, tlocal, plocface, tlocface, corner3d, _, perm = masternodes(porder, 3)

    _, _, _, _, corner2d, _, _ = masternodes(porder, 2)


    for i in [0, 1, 2, 3]:
        print(i)
        a = plocal[corner3d,:][f_idx_template[i,:]]
        b = plocal[perm[:,i][corner2d],:]
        print(a)
        print()
        print(b)
        print(np.all(a==b))
        print()