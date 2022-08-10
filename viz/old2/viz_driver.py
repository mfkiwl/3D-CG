# import external modules
import numpy as np
import os
import pickle
import vis
import logging

logger = logging.getLogger(__name__)

def viz_driver(mesh, sol, fname, call_pv):

    mesh['t'] = mesh['t'].T
    mesh['p'] = mesh['p'].T
    mesh['f'] = mesh['t2f_bdry'].T
    # sol = sol[:,None,:]

    pde = {}
    pde['viselem'] = []
    pde['porder'] = 3
    pde['nd'] = 3
    pde['elemtype'] = 0     # CHANGEME
    pde['paraview'] = 'paraview'
    pde['version'] = 0.3
    mesh['curvedboundaryexpr'] = ''
    mesh['curvedboundary'] = ''

    # pde['visscalars'] = ["potential", 0, 'x0', 4]; # list of scalar fields for visualization
    pde['visscalars'] = ["potential", 0]; # list of scalar fields for visualization
    # pde['visvectors'] = ["potential gradient", np.array([1, 2, 3]).astype(int)]; # list of vector fields for visualization
    pde['visvectors'] = []
    pde['visfilename'] = 'sol_out'

    _ = vis.vis(sol, pde, mesh, fname, call_pv)  # visualize the numerical solution
