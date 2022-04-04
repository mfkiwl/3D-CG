import numpy as np
import pickle

# @profile
def assign_BC_flags(mesh):
    """
    Maps physical groups contained in mesh.pg to the face array in mesh.f
    """

    # Taken from making t2f
    f_idx_template = np.array([[1, 3, 2],    # Nodes on face 0
                               [2, 3, 0],      # Nodes on face 1
                               [0, 3, 1],      # Nodes on face 2
                               [0, 1, 2]])     # Nodes on face 3

    pg_dict = mesh['pg']
    # for pg in pg_dict:
    #     print(pg_dict[pg]['nodes'])
    #     print()

    # Strategy: For each face: all nodes on the face need to be contained in one of the boundaries. We don't have to access the points or the connectivity matrix
    # Subtracting 1 to index the dimension of the face, not the volume element
    nnodes_per_face = mesh['gmsh_mapping'][mesh['elemtype']
                                           ]['nnodes'][mesh['ndim']-1]
    bdry_faces_start_idx = np.where(mesh['f'][:, -1] < 0)[0][0]
    t2f_bdry = np.zeros_like(mesh['t'])

    # Loop through each boundary face
    for iface, face in enumerate(mesh['f'][mesh['f'][:, -1] < 0, :]):
        # Pull the nodes on the face (first three nodes)
        nodes_on_face = face[:nnodes_per_face]#.astype(np.int32)
        # This is the element that has the face on the boundary
        elem = face[nnodes_per_face]

        # This logic was taken from the construction of t2f
        elem1_nodes = mesh['t'][elem, :]
        mask = np.in1d(elem1_nodes, nodes_on_face)

        # Loop through each physical group in the dictionary
        found_flag = False
        for phys_group in pg_dict:
            pg_nodes = pg_dict[phys_group]['nodes']

            # If all the nodes are contained within the pg nodes
            # if np.all([node.tobytes() in pg_nodes for node in nodes_on_face]):  # The dict keys must be of the same datatype (int32)

            if mesh['ndim'] == 2:
                face_bool = (nodes_on_face[0].tobytes() in pg_nodes) and (nodes_on_face[1].tobytes() in pg_nodes)

            elif mesh['ndim'] == 3:
                face_bool = (nodes_on_face[0].tobytes() in pg_nodes) and (nodes_on_face[1].tobytes() in pg_nodes) and (nodes_on_face[2].tobytes() in pg_nodes)

            if face_bool:
                # Assign the boundary flag to the negative boundary ID
                mesh['f'][bdry_faces_start_idx+iface, -1] = - \
                    pg_dict[phys_group]['idx']
                t2f_bdry[elem, :][~mask] = pg_dict[phys_group]['idx']
                found_flag = True

        if not found_flag:
            raise ValueError('Boundary not found for ' + str(nodes_on_face))

        mesh['t2f_bdry'] = t2f_bdry

    return mesh


if __name__ == '__main__':
    print('here')
    with open('boeing_plane_final_processed', 'rb') as file:
        mesh = pickle.load(file)
    assign_BC_flags(mesh)
    print('done')