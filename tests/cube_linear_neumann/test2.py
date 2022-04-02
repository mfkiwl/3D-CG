from functools import reduce
from mpi4py.futures import MPIPoolExecutor

def product(x, y):
    """Return the product of the arguments"""
    return x*y

def sum(x, y):
    """Return the sum of the arguments"""
    return x+y

def funcmpi():
    a = range(1,101)
    b = range(101, 201)

    with MPIPoolExecutor() as executor:
        results = executor.map(product, a, b)

    total = reduce(sum, results)

    print("Sum of the products equals %d" % total)