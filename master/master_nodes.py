import numpy as np

# TODO: extend to account for elements with curved faces and a different porder on the faces (morder)

def master_nodes(porder, ndim):

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

        return plocal, tlocal, plocface, tlocface, permnodes, permedge


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

            # Oveall what we're doing is basically just iterating through all the points in C style format [z, y, x], with [x] changing fastest in the innermost loop. For each x value, compute all the tetrahedra in front of it in y and z.

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
                        if i != ys- 2:
                            tloclist.append(np.array([i, i+1, i+ys, i+zs(npl, j)])+offset)
                            tloclist.append(np.array([i+1+zs(npl, j), i+zs(npl, j), i+ys+zs(npl, j+1), i+ys])+offset)
                            tloclist.append(np.array([i+zs(npl, j), i+1+zs(npl, j), i+1, i+ys])+offset)
                        else:
                            # Last parity 1 element in each X ROW needs to be special - only adding main tet
                            tloclist.append(np.array([i, i+1, i+ys, i+zs(npl, j)])+offset)

                    # Add parity -1 elements - need 3 here
                    for i in np.arange(ys-2):
                        if i != ys - 3:
                            tloclist.append(np.array([i+1+ys, i+ys, i+1, i+1+ys+zs(npl, j+1)])+offset)
                            tloclist.append(np.array([i+ys+zs(npl, j+1), i+1+ys+zs(npl, j+1), i+1+zs(npl, j), i+1])+offset)
                            tloclist.append(np.array([i+1+ys+zs(npl, j+1), i+ys+zs(npl, j+1), i+ys, i+1])+offset)
                        else:
                            # Last parity -1 in each X ROW needs to be special in TYPE II
                            tloclist.append(np.array([i+1+ys, i+1+zs(npl, j), i+ys+zs(npl, j+1), i+ys])+offset)
                            tloclist.append(np.array([i+1+ys, i+1, i+1+zs(npl, j), i+ys])+offset)
                    y_layer_offset += ys
                z_layer_offset += npl

            tlocal = np.asarray(tloclist)

            plocface, tlocface,_,_,_,_ = master_nodes(porder, ndim-1)

            vertex_list = []
            permedge = []
            for i, col in enumerate(plocal.T):
                vertex_list.append(np.where(np.isclose(col,1))[0][0])
                
                # For each column in pl3d, we need to pull out the points that have zeros in them. For points in the same column, they will lie on the same face.
                iface_node_idx = np.where(np.isclose(col, 0))[0]
                permedge.append(iface_node_idx)

            permnodes = np.asarray(vertex_list)
            permface = np.asarray(permedge).T

            # permedge = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}      # Edges on the master tetrahedron
            # for i, pt in enumerate(plocal):
            #     if pt[2] == 0 and pt[3] == 0:   # Along vector [1, 2] in master element
            #         permedge[0].append(i)
            #     elif pt[1] == 0 and pt[3] == 0:   # Along vector [1, 3] in master element
            #         permedge[1].append(i)
            #     elif pt[1] == 0 and pt[2] == 0:   # Along vector [1, 4] in master element
            #         permedge[2].append(i)
            #     elif pt[0] == 0 and pt[3] == 0:   # Along vector [2, 3] in master element
            #         permedge[3].append(i)
            #     elif pt[0] == 0 and pt[2] == 0:   # Along vector [2, 4] in master element
            #         permedge[4].append(i)
            #     elif pt[0] == 0 and pt[1] == 0:   # Along vector [3, 4] in master element
            #         permedge[5].append(i)

            # permedge = np.asarray(permedge).T

            permedge = np.zeros((porder+1, 6))
                        # 1->2 1->3 1->4 2->3       3->4                            4->2
            permedge[0,:] = [0, 0, 0, porder, int((porder+1)*(porder+2)/2-1), int((porder+1)*(porder+2)*(porder+3)/6-1)]
            for i in np.arange(1, porder+1):
                permedge[i, 0] = i      # 1->2
                permedge[i, 1] = permedge[i-1, 1] + porder+2-i      # 1->3
                permedge[i, 2] = permedge[i-1, 2] + int((porder+2-i)*(porder+3-i)/2)        # 1->4
                permedge[i, 3] = permedge[i-1, 3] + porder+1-i      # 2->3
                permedge[i, 4] = permedge[i-1, 4] + int((porder+1-i)*(porder+2-i)/2)           # 3->4
                permedge[i, 5] = permedge[i-1, 5] - int(i*(i+1)/2) - i+2           # 4->2

            # Since the parametric coordinates have served their purpose, remove them (the first column) of the local point matrices while returning
            plocal = plocal[:, 1:]

    return plocal, tlocal, plocface, tlocface, permnodes, permedge, permface


if __name__ == '__main__':
    mkmshlocal(porder=3)