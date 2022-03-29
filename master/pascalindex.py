import numpy as np

def pascalindex2d(k):
    """
    The purpose of this function is to prepare a set of combinations of m and n for the creation of the Jacobi polynomials that satisfy (m+n)<=k.
    k: int, max order of the Jacobi polynomial desired

    Returns: array of (ncomb x 2), where each row corresponds to a tuple of (m, n) values for the Jacobi polynomials. Take, for example, the monomial basis: {1, x1, x2, x1^2, x1x2, x2^2, ...}. The order of the polynomials in the variables x1 and x2 are (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), which is exactly what this function outputs.
    Remember the formula nplocal = (k+1)(k+2)/2 for 2D? This is just a formula for the number of elements up to that order (row) of the pascal triangle. Assuming each entry has equal area weighting, this formula basically calculates the "area" of the elements in pascal's triangule using A=(base1*base2)/2*(1/3)h - volume of a square pyramid is (1/3)bh.
    """

    if k==0:
        # Base case - top element in pascal triangle has index (0, 0)
        return np.array([0, 0])[None, :]

    # Builds the index pairs of the kth row of the triangle

    nth_row = np.zeros((k+1, 2))
    nth_row[:, 0] = np.arange(k+1)
    nth_row[:, 1] = k - nth_row[:, 0]

    nth_row = np.fliplr(nth_row)    # So it matches the 16.930 matlab code
    # And then stacks it with the index pairs of the elements in the rows of the pyramid above it
    prev_rows = pascalindex2d(k-1)
    pindx = np.concatenate((prev_rows, nth_row), axis=0).astype(np.int)

    return pindx


def pascalindex3d(k):
    """
    The purpose of this function is to prepare a set of combinations of m, n, and l for the creation of the Jacobi polynomials that satisfy (m+n+l)<=k.
    k: int, max order of the Jacobi polynomial desired

    Returns: array of (ncomb x 3), where each row corresponds to a tuple of (m, n, l) values for the Jacobi polynomials.
    Remember the formula nplocal = (k+1)(k+2)(k+3)/6 for 3D? This is just a formula for the number of elements up to that order (row) of the pascal pyramid. Assuming each entry has equal volume weighting, this formula basically calculates the "volume" of the elements in pascal's triangule using A=bh/2*(altitude)/3
    Recursively 
    """

    if k==0:
        # Base case - top element in pascal pyramid has index (0, 0, 0)
        return np.array([0, 0, 0])[None,:]

    # Builds the index pairs of the kth level of the pyramid
    level2d_indices = pascalindex2d(k)  # 2D pascal indices on a level
    nth_level = np.zeros((level2d_indices.shape[0], 3))
    nth_level[:,1:] = level2d_indices
    nth_level[:, 0] = k-np.sum(nth_level[:,1:], axis=1)   # Adds the third column/dimension so that the rows add to k

    # nth_level = np.fliplr(nth_level)    # So it matches the 16.930 matlab code
    # And then stacks it with the index pairs of the elements in the levels of the pyramid above it
    prev_levels = pascalindex3d(k-1)
    pindx = np.concatenate((prev_levels, nth_level), axis=0).astype(int)

    return pindx

if __name__ == '__main__':
    print(pascalindex2d(3))
    # print(pascalindex3d(3))