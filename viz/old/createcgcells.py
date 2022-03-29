from numpy import *

def  createcgcells(cgelcon,tlocal,ne):
    # Constructs a connectivity matrix into the set of *CG* nodes for each DG-style element - this would be of dimension (numel x nplocal x numpts_per_elem)

    nce, nve = tlocal.shape;
    cells = zeros((ne*nce,nve));
    tlocal = tlocal.flatten('F')-1;
    for el in range (0,ne):
        m = nce*el;
        cells[m:(m+nce),:] = reshape(cgelcon[tlocal,el],[nce,nve],'F');
    #cells = cells - 1;

    return cells
