import numpy as np

def gmshwrite(p, t, fname, f=None):
    with open(fname + '.msh', 'w') as file:
        file.write('$MeshFormat\n3 0 8\n$EndMeshFormat\n$Nodes\n')
        file.write(str(p.shape[0])+'\n')
        for i, row in enumerate(p):
            file.write('{:d} {:f} {:f} {:f} 0\n'.format(i+1, row[0], row[1], row[2]))
        file.write('$EndNodes\n$Elements\n')

        # Writing the element numbers
        if f is not None:
            file.write(str(t.shape[0]+f.shape[0])+'\n')
        else:
            file.write(str(t.shape[0])+'\n')

        # Writing the elements
        for i, row in enumerate(t+1):
            file.write('{:d} 4 {:d} 4 {:d} {:d} {:d} {:d}\n'.format(i+1, 0, row[0], row[1], row[2], row[3]))
        if f is not None:
            for i, row in enumerate(f):
                if row[-1]<0: # boundary face
                    file.write('{:d} 2 {:d} 3 {:d} {:d} {:d}\n'.format(t.shape[0]+i+1, -row[-1], row[0]+1, row[1]+1, row[2]+1))
                else:
                    file.write('{:d} 2 {:d} 3 {:d} {:d} {:d}\n'.format(t.shape[0]+i+1, 0, row[0]+1, row[1]+1, row[2]+1))                
        file.write('$EndElements')
