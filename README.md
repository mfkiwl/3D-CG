Curved boundaries not implemented (only polygonal boundaries allowed)
No support for master/nonuniformlocalpts -> you can't map the points to a nonuniform (non-affine) transformation

All the points and elements are 0-indexed, except for:
-f, which is 1-indexed (need to be able to represent + and - faces to indicate direction)

To verify python vs matlab output
In matlab: 'save 'varname.mat' varname' - make sure the varnames match!
Copy the .mat file to the directory where the python script is being tested
In script:
    import sys
    sys.path.insert(0, '../util')
    from import_util import load_mat
    varname_mat = load_mat('varname')
    print(np.allclose(varname, varname_mat))
    
    If you're comparing a python 3D array with a 3D array imported from matlab using scipy.io.loadmat, do
    varname_mat = load_mat('varname').transpose((2, 0, 1))
    
np print options: 
    np.set_printoptions(suppress=True, linewidth=np.inf)
    # np.set_printoptions(linewidth=np.inf, precision=10)
    np.set_printoptions(suppress=True, linewidth=np.inf, precision=4)

Whenever trying to reshape a np array like a matlab array, always do something like this: ph = np.ravel(ph, order='F').reshape((-1, 2))
You need to ravel the array with Fortran ordering to get back to the base-level 1D vector, then reshape that using C-ordering which is the default for reshape. This decouples the ordering of the raveling and reshaping processes, which ultimately both happen for one call to reshape
https://stackoverflow.com/questions/45973722/how-does-numpy-reshape-with-order-f-work

https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
Also from the docs: You can think of reshaping as first raveling the array (using the given index order), then inserting the elements from the raveled array into the new array using the same kind of index ordering as was used for the raveling.

My mistake in reshaping the dgnodes array was that I was assuming that the raveled array was then reshaped with C ordering, when it was really reshaped with the same ordering that was passed to reshape, which was less flexible than the combo ravel/reshape.

np.transpose is also used to permute dimensions too.

Example of raising an exception:
    elif dim == 3:
        raise NotImplementedError('Dimension 3 not implemented yet')

List of exceptions to use:
https://docs.python.org/3/library/exceptions.html#exception-hierarchy

Put all conversion to 1-indexing after the output of the function, in the testing section

Using np.allclose: if you are comparing something to machine precision, you HAVE to READ THE DOCUMENTATION!!!
I have been burned in the past by not using the correct tolerances. Using the default values will not catch errors on the order of 10e-7!!
To compare something close to machine precision, use something like np.allclose(bad, mat_complete, rtol=1e-13, atol=4e-15) and then tweak as necessary if the bounds are too tight.
If finding the unique elements of a float array and need to round the inputs like using "snap", DON'T USE THE OUTPUT SORTED ARRAY! USE THE SORTED INDICES FROM NP.UNIQUE AND INDEX INTO THE ORIGINAL (NON-ROUNDED) ARRAY

Order of operations

Clean up the current code - comment, github, remove temp files for testing
Vectorize jacobians and koornwinder jacobian matrix, make it a bit faster
Switch to conjugate gradient for the solver
Take out the intermediate ae and fe data structures
Fix assignment of boundary conditions
Convert to 3D - make a new directory for this


matlab -softwareopengl


The link between the orientation/position of the element in the mesh as it is known to the sovler, and the list of nodes, is done via the connectivity matrix and the affine transformation. The assumption/enabler behind this abstraction is that the nodes for every single element are given in the connectivity matrix in the same relative order to each other in the mesh. For example, for a 2D mesh it is assumed that the nodes are ALWAYS named going CCW. It doesn't matter which of the three nodes is the starting node (the connectivity matrix could have been rolled for all we care), but what is crucial is that the element nodes are all assigned going in the same order. 

It can be assumed that the order is the same for all elements, so it honestly doesn't even matter that they conform to one orientation or the other (CCW vs CW in the case of 2D elements). The good thing about using simplex elements is that if you choose ndim points, you automatically define a face, and there is only 1 additional node left, which is the only choice for the last node. Thus, there is no ambiguity in which node is selected.

Truly, the only imortant thing is that the node numbering is consistent for all the elements.

In 2D: faces were the between adjacent nodes, and you rolled from node3->node1 to get the third face. There was only one ambiguity as to the convention taken, and that was the traversal orientation of the triangle.
In 3D: faces are a bit more complicated, but there is also only one ambiguity: the traversal orientation of a single face of the triangle. The first three nodes specify a triangle, and then the fourth node is automatically defined.

There is choice in the mesh generator as to how to orient the face. I assume the triangular faces are traversed CCW but should check that. Might want to go into gmsh to see how the nodes are numbered

In a more general case, define a rule for face node ordering on the element.
In 2D, this was:
1,2
2,3
3,1

This rule was easily implemented by rolling the mesh.t data structure. In 3D this is slightly more complicated but follows the same general idea:
1,2,3
1,2,4
1,3,4
2,4,3

And of course, these indices may be rolled within their own faces too (like 3,4,2 for the last one). This is like how an extra column was rolled over in the generation of the 2D mkt2t. For the same face, you could have
2,4,3
4,3,2
3,2,4

So the complete list would look like 2,4,3,2,4 to capture all of the combos. But then you need to exclude the other matches/repeats that may be throwing you off.


Actually judging from here: http://victorsndvg.github.io/FEconv/formats/gmshmsh.xhtml, it looks like gmsh goes CCW around the first face VIEWED FROM THE OPPOSITE NODE. This is okay - I will take the face numberings such that the nodes on the ith face opposite the ith element are going CCW around the triangle.So the faces would then be:
So, the method for obtaining the faces becomes (more systematic): for each node in the element, exclude node i and then the other 3 nodes form a face. For a tet we have:
2,4,3
3,4,1
1,4,2
1,2,3


CONVENTION FOR MESH.T2T 3D
For the ith node, list the element across from that node that shares the ith face.

CONVENTION FOR MESH.F 3D
Standard for the faces in mesh.f: Traversing the face in increasing order of the node numbers, take the normal vector (3D) of the triangle going around. The element that the normal vector points into will be listed first, and the other element will be listed second.

If the element is on the boundary, the order of the nodes on the face doesn't matter, and the boundary element is listed first, followed by a -bdry_nbr

CONVENTION FOR MESH.T2F 2D:
For each node, list the face across from it. If the face is traversed CCW, then the face number is positive. If not, it is negative.

Each data structure takes out a degree of freedom, for example, it was arbitrarily chosen to traverse the faces in mesh.f in order of increasing node number. Thus, we needed to introduce a new data structure, mesh.t2f, that performs the mapping to take out that ambiguity.


Ex: from h1.0_tets24.msh, element 38 has nodes 12, 7, 3, 13
