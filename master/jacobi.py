from scipy.special import gamma
import numpy as np
from numpy.polynomial import Polynomial as poly
fact = np.math.factorial


def nCr(n, r):
    return gamma(n+1)/(gamma(r+1)*gamma(n-r+1))

def upper_pochhammer(qty, n):
    result = 1
    for i in np.arange(n):
        result *= qty+i
    return result

def jacobi_recursive(n, a, b):
    """
    From 16.930 Project 1 handout

    2(n + 1)*(n + α + β + 1)*(2n + α + β)*P_{n+1}^{α, β}(x) = [(2n + α + β + 1)*(α^2 − β^2) + (2n + α + β)_3*x]*P{α, β}_n(x) - 2(n + α)*(n + β)*(2n + α + β + 2)*P_{n−1}^{α,β}(x)
    """

    # Base cases
    if n == 0:
        return poly([1])
    elif n == 1:
        return poly([(a-b)/2, (a+b+2)/2])

    n -= 1  # The formulas are for the n+1th polynomial, so here the nth polynomial really means the one with degree one less

    # Prepare coefficients
    c1 = 2*(n+1)*(n+a+b+1)*(2*n+a+b)
    c2 = (2*n + a + b + 1)*(a**2 - b**2)
    c3 = (2*n + a + b)*(2*n + a + b + 1)*(2*n + a + b + 2)
    c4 = 2*(n+a)*(n+b)*(2*n+a+b+2)

    tmp_poly = poly([c2, c3])
    p_n = jacobi_recursive(n, a, b)
    p_n_minus1 = jacobi_recursive(n-1, a, b)

    # Note: output is in the reverse order of the 16.930 matlab code
    return (tmp_poly*p_n - c4*p_n_minus1)/c1

def jacobi_wiki_realx(n, a, b):
    """
    From wikipedia (https://en.wikipedia.org/wiki/Jacobi_polynomials#Alternate_expression_for_real_argument): For real x, no restriction on n, a, or b

    """
    poly1 = poly([-1, 1])/2
    poly2 = poly([1, 1])/2

    result = poly([0])
    for s in np.arange(n+1):
        summand = nCr(n+a, n-s) * nCr(n+b, s)*poly1**s * poly2**(n-s)
        result += summand

    return result


def jacobi_wiki_nat_numbers(n, a, b):
    """
    From wikipedia (https://en.wikipedia.org/wiki/Jacobi_polynomials#Alternate_expression_for_real_argument): special case that n, a, and beta are nonnegative integers

    """
    coeff = fact(n+a)*fact(n+b)

    poly1 = poly([-1, 1])/2
    poly2 = poly([1, 1])/2
    
    result = poly([0])
    for s in np.arange(n+1):
        denom = fact(s)*fact(n+a-s)*fact(b+s)*fact(n-s)
        summand = (1/denom) * poly1**(n-s) * poly2**(s)
        result += summand
    
    return result*coeff

def jacobi(k, a, b):
    """
    From https://arxiv.org/pdf/2105.07547.pdf

    This is the one that I've selected to use going forward
    """
    poly1 = poly([-1, 1])/2

    result = poly([0])
    for j in np.arange(k+1):
        coeff = upper_pochhammer(a+j+1, k-j)*upper_pochhammer(k+a+b+1, j)/(fact(j)*fact(k-j))
        summand = coeff* poly1**j
        result += summand

    return result

if __name__ == '__main__':
    n = 3
    a = 0
    b = 0
    print(jacobi_recursive(n, a, b))
    print(jacobi_wiki_nat_numbers(n, a, b))
    print(jacobi_wiki_realx(n, a, b))
    print(jacobi(n, a, b))
