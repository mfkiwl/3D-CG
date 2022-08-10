import os, sys, shutil
from numpy import *
from createcggrid import createcggrid
from vtuwrite import vtuwrite
from master_nodes import masternodes
from createdgnodes import createdgnodes
from mkshape import mkshape
import logging

logger = logging.getLogger(__name__)

def vis(visfields, app, mesh, fname, call_pv):

    if app['viselem'] == []:
        ne = mesh['t'].shape[1];
        app['viselem'] = range(0,ne);

    if app['porder']>1:
        visorder = min(2*app['porder'],8);
    else:
        visorder = app['porder'];

    visorder=8

    mesh['xpe'] = masternodes(app['porder'],app['nd'],app['elemtype'])[0];
    xpe,telem = masternodes(visorder,app['nd'],app['elemtype'])[0:2];

    visshape = mkshape(app['porder'],mesh['xpe'],xpe,app['elemtype']);
    visshape = visshape[:,:,0].T;

    dgnodes = createdgnodes(mesh['p'],mesh['t'][:,app['viselem']],mesh['f'][:,app['viselem']],mesh['curvedboundary'],mesh['curvedboundaryexpr'],visorder);
    cgnodes, cgelcon, cgcells, celltype = createcggrid(dgnodes,telem)[0:4];
    dgnodes = createdgnodes(mesh['p'],mesh['t'][:,app['viselem']],mesh['f'][:,app['viselem']],mesh['curvedboundary'],mesh['curvedboundaryexpr'],app['porder']);

    tm = matmul(visshape,reshape(visfields[:,:,app['viselem']],(visshape.shape[1], visfields.shape[1]*len(app['viselem'])), 'F'));
    tm = reshape(tm,(visshape.shape[0], visfields.shape[1], len(app['viselem'])),'F');

    print(tm)
    vtuwrite(fname, cgnodes, cgelcon, cgcells, celltype, app['visscalars'], app['visvectors'], tm);

    logging.info('Wrote .VTU to '+fname)
    if call_pv:
        str = "paraview --data=" + fname + ".vtu &"
        logging.info('Calling paraview!')
        os.system(str);

    return dgnodes;
