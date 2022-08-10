import os, sys, shutil
from numpy import *
from vtuwrite import vtuwrite
import os

def getcelltype(nd, elemtype):
    if nd == 2:
        if elemtype == 0:
            cell_t = 5
        else:
            cell_t = 9
    elif nd == 3:
        if elemtype == 0:
            cell_t = 10
        else:
            cell_t = 12
    else:
        sys.exit("Number of mesh spatial dimensions should be 2 or 3")
    return cell_t

def createcgcells(cgelcon, tlocal, ne):
    # Constructs a connectivity matrix into the set of *CG* nodes for each DG-style element - this would be of dimension (numel x nplocal x numpts_per_elem)
    nce, nve = tlocal.shape
    cells = zeros((ne*nce, nve))
    tlocal = tlocal.flatten('F')
    for el in range(0, ne):
        m = nce*el
        cells[m:(m+nce), :] = reshape(cgelcon[tlocal, el], [nce, nve], 'F')
    return cells

def createcggrid(mesh, elemtype=0):
    # Get some dimensions of the mesh
    nd = mesh['dgnodes'].shape[1]
    ne = mesh['dgnodes'].shape[2]
    cgcells = createcgcells(mesh['tcg'],mesh['tlocal'],ne)
    cell_t = getcelltype(nd,elemtype)
    return cgcells, cell_t

def vis(visfields,visscalars, visvectors, mesh, master):
    # Uses the same polynomial order of approximation for the visualization
    viz_filename = 'viz_test'
    numel = mesh['t'].shape[0]
    visshape = master.shap[:,:,0]

    cgcells, celltype = createcggrid(mesh)
    tm = matmul(visshape,reshape(visfields,(visshape.shape[1], visfields.shape[1]*numel), 'F'))
    tm = reshape(tm,(visshape.shape[0], visfields.shape[1], numel),'F')

    vtuwrite(viz_filename, mesh['pcg'], mesh['tcg'], cgcells, celltype, visscalars, visvectors, tm)
    os.system("paraview --data=" + viz_filename + ".vtu &")
