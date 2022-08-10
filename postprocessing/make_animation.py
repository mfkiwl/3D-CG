import numpy as np
import pyvista as pv
import field_orientation
import pickle


def make_frames(root):
    vtu_dirname = '/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/{}/'.format(root)
    vtu_fname = '{}_surface.vtu'.format(root)
    surf_soln = pv.read(vtu_dirname + vtu_fname)

    angle_vec = np.linspace(0, 2*np.pi, num=100, endpoint=False)
    angles = np.zeros((angle_vec.shape[0], 3))
    angles[:,0] = angle_vec

    angles = np.concatenate((angles, np.roll(angles, 1, axis=1), np.roll(angles, 2, axis=1)), axis=0)

    for i, angle_tup in enumerate(angles):
        print(i)
        alpha, beta, gamma = angle_tup
        
        if alpha > 0:     # Hard-coding euler angles
            x_fac = np.cos(alpha)
            y_fac = 0
            z_fac = np.sin(alpha)

        elif beta > 0:     # Hard-coding euler angles
            x_fac = 0
            y_fac = np.cos(beta)
            z_fac = np.sin(beta)

        elif gamma > 0:     # Hard-coding euler angles
            x_fac = np.cos(gamma)
            y_fac = np.sin(gamma)
            z_fac = 0
        else:
            x_fac = 1
            y_fac = 0
            z_fac = 0

        # print(alpha, beta, gamma)
        # print(x_fac, y_fac, z_fac)
        # print()
        data = surf_soln.copy()
        data.clear_data()
        data.point_data['e_dot_n'] = x_fac * surf_soln.point_data['Ex_Surface Normal Electric Field'] + y_fac * surf_soln.point_data['Ey_Surface Normal Electric Field'] + z_fac * surf_soln.point_data['Ez_Surface Normal Electric Field']
        data.save('/media/homehd/saustin/lightning_research/3D-CG/postprocessing/fem_solutions/{}/frames/frame{}.vtu'.format(root, i))

if __name__ == '__main__':
    # root = 'blackhawk'
    # root = 'd8'
    root = 'bwb'
    make_frames(root)

    # #!/bin/bash

    # ffmpeg -r 20 -i frame.%04d.png -vcodec libx264 -crf 25 -vcodec libx264 -an d8_E_field_sweep.mp4