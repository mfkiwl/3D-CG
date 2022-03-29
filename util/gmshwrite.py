import numpy as np

def gmshwrite(t, p, f, porder, fname):
    with open(fname + '.msh', 'w') as file:
        file.write('$MeshFormat\n3 0 8\n$EndMeshFormat\n$Nodes\n')
        file.write(str(p.shape[0])+'\n')
        for i, row in enumerate(p):
            file.write('{:d} {:f} {:f} {:f} 0\n'.format(i+1, row[0], row[1], row[2]))
        file.write('$EndNodes\n$Elements\n')
        if f is not None:
            file.write(str(t.shape[0]+f.shape[0])+'\n')
        else:
            file.write(str(t.shape[0])+'\n')
        for i, row in enumerate(t+1):
            file.write('{:d} 4 {:d} 4 {:d} {:d} {:d} {:d}\n'.format(i+1, i+1, row[0], row[1], row[2], row[3]))
        for i, row in enumerate(f+1):
            file.write('{:d} 2 {:d} 3 {:d} {:d} {:d}\n'.format(t.shape[0]+i+1, i+1, row[0], row[1], row[2]))
            
        file.write('$EndElements')
