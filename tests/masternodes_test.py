import numpy as np
# Finding the sim root directory
import sys
from pathlib import Path
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('master')))
import masternodes

def masternodes_perm_test(porder):
    dim=3
    plocal, tlocal, plocface, tlocface, corner3d, _, perm = masternodes.masternodes(porder, dim)

    _, _, _, _, corner2d, _, _ = masternodes.masternodes(porder, 2)

    f_idx_template = np.array([[1, 3, 2],    # Nodes on face 0
                               [2, 3, 0],      # Nodes on face 1
                               [0, 3, 1],      # Nodes on face 2
                               [0, 1, 2]])     # Nodes on face 3

    face_match_bool = np.full((f_idx_template.shape[0]), False, dtype=bool)

    for i, _ in enumerate(face_match_bool):
        a = plocal[corner3d,:][f_idx_template[i,:]]
        b = plocal[perm[:,i][corner2d],:]
        face_match_bool[i] = np.all(a==b)

    return np.all(face_match_bool)

if __name__ == '__main__':
    print(masternodes_perm_test(3))