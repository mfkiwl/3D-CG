from numpy import *
from koornwinder1d_es import *
from koornwinder2d_es import *
from koornwinder3d_es import *

def koornwinder_es(x,porder):

    n = 0
    dim = 0
    if len(x.shape)==1:
        n = x.shape[0]
        dim = 1
        x = reshape(x,(n,dim))
    else:
        n = x.shape[0]
        dim = x.shape[1]
        x = reshape(x,(n,dim))

    if dim==1:
        f, fx, fy, fz = koornwinder1d_es(x, porder)
    elif dim==2:
        f, fx, fy, fz = koornwinder2d_es(x, porder)
    elif dim==3:
        f, fx, fy, fz = koornwinder3d_es(x, porder)

    return f,fx,fy,fz
