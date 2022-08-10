import matplotlib
import numpy as np
import pickle
from pathlib import Path
import sys
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from matplotlib import colors
from sympy import rotations

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Finding the sim root directory
cwd = Path.cwd()
for dirname in tuple(cwd.parents):
    if dirname.name == '3D-CG':
        sim_root_dir = dirname
        continue

sys.path.append(str(sim_root_dir.joinpath('util')))
sys.path.append(str(sim_root_dir.joinpath('mesh')))
sys.path.append(str(sim_root_dir.joinpath('master')))
sys.path.append(str(sim_root_dir.joinpath('viz')))
sys.path.append(str(sim_root_dir.joinpath('CG')))
sys.path.append(str(sim_root_dir.joinpath('logging')))
import viz

def plot_sphere(theta, phi, val_mat):
    # print(theta.shape)
    # print(phi.shape)
    # print(val_mat.shape)

    # #theta inclination angle
    # #phi azimuthal angle
    # # n_theta = 50 # number of values for theta
    # # n_phi = 200  # number of values for phi
    # # r = 2        #radius of sphere

    # # theta, phi = np.mgrid[0.0:0.5*np.pi:n_theta*1j, 0.0:2.0*np.pi:n_phi*1j]



    # # plot_sphere(phi_mat*np.pi/180, theta_mat*np.pi/180, baseline_attach1_zones_mat)
    # # theta: 0 -> pi
    # # phi: 0 -> 2*pi
    # r=1
    # x = r*np.sin(theta)*np.cos(phi)
    # y = r*np.sin(theta)*np.sin(phi)
    # z = r*np.cos(theta)

    # # mimic the input array
    # # array columns phi, theta, value
    # # first n_theta entries: phi=0, second n_theta entries: phi=0.0315..
    # # inp = []
    # # for j in phi[0,:]:
    # #     for i in theta[:,0]:
    # #         val = 0.7+np.cos(j)*np.sin(i+np.pi/4.)# put something useful here
    # #         inp.append([j, i, val])
    # # inp = np.array(inp)
    # # print(inp.shape)
    # # print(inp[49:60, :])

    # # #reshape the input array to the shape of the x,y,z arrays. 
    # # c = inp[:,2].reshape((n_phi,n_theta)).T
    # # print(z.shape)
    # # print(c.shape)

    # val_mat = np.ones_like(val_mat)
    # print(x.shape)
    # print(y.shape)
    # print(val_mat.shape)

    # # cmap = plt.get_cmap('tab20', 7)
    # # color_vals = cmap[val_mat]
    # face_colors = cm.tab20(val_mat)

    # print(face_colors.shape)
    # # exit()

    # #Set colours and render
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # #use facecolors argument, provide array of same shape as z
    # # cm.<cmapname>() allows to get rgba color from array.
    # # array must be normalized between 0 and 1
    # ax.plot_surface(x,y,z,  rstride=1, cstride=1, facecolors=face_colors, alpha=1, linewidth=1)
    # ax.set_xlim([-2.2,2.2])
    # ax.set_ylim([-2.2,2.2])
    # ax.set_zlim([0,4.4])
    # # ax.set_aspect()
    # #ax.plot_wireframe(x, y, z, color="k") #not needed?!
    # # plt.savefig(__file__+".png")
    # plt.show()

    #theta inclination angle
    #phi azimuthal angle
    n_theta = 120 # number of values for theta
    n_phi = 241  # number of values for phi
    r = 1        #radius of sphere

    theta, phi = np.mgrid[0.0:0.5*np.pi:n_theta*1j, 0.0:2.0*np.pi:n_phi*1j]

    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)

    # mimic the input array
    # array columns phi, theta, value
    # first n_theta entries: phi=0, second n_theta entries: phi=0.0315..
    # inp = []
    # for j in phi[0,:]:
    #     for i in theta[:,0]:
    #         val = 0.7+np.cos(j)*np.sin(i+np.pi/4.)# put something useful here
    #         inp.append([j, i, val])
    # inp = np.array(inp)
    # print(inp.shape)
    # print(inp[49:60, :])

    #reshape the input array to the shape of the x,y,z arrays. 
    # c = inp[:,2].reshape((n_phi,n_theta)).T
    print(z.shape)
    # print(c.shape)
    # facecolors=cm.hot(c/c.max())
    # color_vals = cmap[val_mat]
    cmap = plt.get_cmap('tab20', 7)
    facecolors = cmap(val_mat)
    print(np.unique(facecolors, return_counts=True))
    print(facecolors.shape)

    m = cm.ScalarMappable(cmap=cmap)
    m.set_array(val_mat)

    #Set colours and render
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    #use facecolors argument, provide array of same shape as z
    # cm.<cmapname>() allows to get rgba color from array.
    # array must be normalized between 0 and 1
    ax.plot_surface(
        x,y,z,  rstride=1, cstride=1, facecolors=facecolors, alpha=1, linewidth=1)
    ax.set_xlim([-1.2,1.2])
    ax.set_ylim([-1.2,1.2])
    ax.set_zlim([0,2.4])
    cbar = plt.colorbar(m, ax=ax, aspect=50)
    #ax.plot_wireframe(x, y, z, color="k") #not needed?!
    # plt.savefig(__file__+".png")
    plt.show()


def get_cmap_vals(cmap_id, num):
    print('Colormap values for {}, {} pts'.format(cmap_id, num))
    cmap = plt.get_cmap(cmap_id, num)
    for i in range(cmap.N):
        rgba = cmap(i)
        print(i, rgba[:-1]) # Chopping off a value in rgba

def local_attachment_postprocessing(summaries_aggregate_dname, sol_fname, case):

    with open(sol_fname, 'rb') as file:
        solution = pickle.load(file)

    # Read from disk
    with open('{}baseline_attach_pt1_mat.npy'.format(summaries_aggregate_dname), 'rb') as file:
        baseline_attach_pt1_mat = np.load(file).astype(np.int64)
    with open('{}baseline_attach_pt2_mat.npy'.format(summaries_aggregate_dname), 'rb') as file:
        baseline_attach_pt2_mat = np.load(file).astype(np.int64)
    with open('{}baseline_leader1_sign_mat.npy'.format(summaries_aggregate_dname), 'rb') as file:
        baseline_leader1_sign_mat = np.load(file).astype(np.int64)
    with open('{}baseline_leader2_sign_mat.npy'.format(summaries_aggregate_dname), 'rb') as file:
        baseline_leader2_sign_mat = np.load(file).astype(np.int64)
    with open('{}baseline_Eattach_mat.npy'.format(summaries_aggregate_dname), 'rb') as file:
        baseline_Eattach_mat = np.load(file)

    with open('{}q_opt_pos_attach_mat.npy'.format(summaries_aggregate_dname), 'rb') as file:
        q_opt_pos_attach_mat = np.load(file).astype(np.int64)
    with open('{}q_opt_neg_attach_mat.npy'.format(summaries_aggregate_dname), 'rb') as file:
        q_opt_neg_attach_mat = np.load(file).astype(np.int64)
    with open('{}q_opt_Eattach_mat.npy'.format(summaries_aggregate_dname), 'rb') as file:
        q_opt_Eattach_mat = np.load(file)
    with open('{}q_opt_mat.npy'.format(summaries_aggregate_dname), 'rb') as file:
        q_opt_mat = np.load(file)
    with open('{}theta_mat.npy'.format(summaries_aggregate_dname), 'rb') as file:
        theta_mat = np.load(file)
    with open('{}phi_mat.npy'.format(summaries_aggregate_dname), 'rb') as file:
        phi_mat = np.load(file)

    surf_mesh = solution['surf_mesh']

    attach_p1_flattened = baseline_attach_pt1_mat.ravel()
    attach_p2_flattened = baseline_attach_pt2_mat.ravel()
    # print(np.unique(attach_p1_flattened))
    # print(np.unique(attach_p2_flattened))

    # Visualize a paraview file and ask the user to input the number of attachment regions that they see
    viz_bool = True
    if viz_bool:
        surf_attach_pts = np.zeros((surf_mesh['pcg'].shape[0], 1))
        surf_attach_pts[attach_p1_flattened,:] = 1   # Need to broadcast to 2D array
        surf_attach_pts[attach_p2_flattened,:] = 1
        viz.visualize(solution['surf_mesh'], 2, {'scalars':{0: 'Attachment points'}}, 'attachment_points', True, surf_attach_pts, None, type='surface_mesh') # Can only have scalars on a surface mesh

        print('In the ParaView window, look at the aircraft with the distribution of attachment points. Enter, as a whole number, the number of attachment zones that you see and press Enter:')
        num_attach_zones = int(input())

    else:
        # Default number of attachment zones without opening paraview
        num_attach_zones = 24

    # Using the baseline attachment points (combined positive and negative), use k-means clustering to identify the attachment zones
    baseline_attach_pts = np.concatenate((attach_p1_flattened, attach_p2_flattened))    # No axis argument, both 1D arrays

    baseline_attach_coords = surf_mesh['pcg'][baseline_attach_pts, :]

    attachment_zones = KMeans(n_clusters=num_attach_zones, init='k-means++', n_init=20, max_iter=300, random_state=0, algorithm='elkan').fit(baseline_attach_coords)
    attachment_zones.labels_ += 1

    baseline_attach1_zones_mat = attachment_zones.labels_[:attach_p1_flattened.shape[0]].reshape(baseline_attach_pt1_mat.shape)     # These two are zero indexed
    baseline_attach2_zones_mat = attachment_zones.labels_[attach_p2_flattened.shape[0]:].reshape(baseline_attach_pt2_mat.shape)

    classified_attach_pts = np.zeros((surf_mesh['pcg'].shape[0], 1))
    classified_attach_pts[baseline_attach_pts,:] = attachment_zones.labels_[:,None]
    convert_idx_vec = np.zeros(np.max(attachment_zones.labels_)+1, dtype=np.int)    # Need to convert to 1 indexing

    # viz_classify_bool = True
    viz_classify_bool = False
    if viz_classify_bool:
        viz.visualize(solution['surf_mesh'], 2, {'scalars':{0: 'Attachment points'}}, 'attachment_points', True, classified_attach_pts, None, type='surface_mesh') # Can only have scalars on a surface mesh
        exit()
        print('In the ParaView window, look at the aircraft with the distribution of attachment points. Enter, in sets of numbers separated by spaces, the indices for each group. For example: 4 8 2. Press Enter to submit each group and press Enter again when done:')
        
        # stop_bool = False
        # zones_list = []
        # while not stop_bool:
        #     input_str = input()
        #     if input_str == '':
        #         stop_bool = True
        #         continue
        #     clusters_list = input_str.split()
        #     for i, val in enumerate(clusters_list):
        #         clusters_list[i] = int(val)
        #     zones_list.append(clusters_list)

    # D8
    # zones_list = [[4, 11, 13, 17, 18, 19, 20, 21, 23], [2, 14, 22], [3, 10, 24, 7], [16, 5], [1, 12, 9], [8, 15], [6]]
    
    # E190_430K
    zones_list = [[1, 24, 16, 23, 17], [3, 10, 15, 13, 20], [4, 12, 9, 21], [5, 22, 7], [14], [8, 19, 2], [6, 18, 11]]
    
    # BWB
    # zones_list = [[6, 7, 12, 10, 15, 18, 17, 3, 19], [24, 13, 4, 20], [2, 21, 23, 11, 8] , [1, 14, 5, 9, 16, 22]]

    num_zones = len(zones_list)
    attachment_probabilities = {}

    tot_num_attach = 2*baseline_attach_pt1_mat.size     # Total number of baseline attachment points
    count_arr = np.bincount(attachment_zones.labels_)

    zone_id_array = np.zeros((surf_mesh['pcg'].shape[0], 1))
    # Classify main attachment groups and their probabilities
    for izone, zone in enumerate(zones_list):
        izone += 1     # 1-indexing
        zone = np.array(zone)
        # Compute centroid of clusters and associate zone center with nearest point
        zone_centroid = np.mean(attachment_zones.cluster_centers_[zone-1,:], axis=0)
        # print(zone_centroid)

        attach_pts_mask = np.zeros(surf_mesh['pcg'].shape[0], dtype=bool)
        pt_idx_in_zone = np.arange(surf_mesh['pcg'].shape[0])
        for cluster in zone:
            convert_idx_vec[cluster]=izone
            attach_pts_mask[baseline_attach_pts[np.argwhere(attachment_zones.labels_ == cluster).ravel()]] = True

        coords_in_zone = surf_mesh['pcg'][attach_pts_mask,:]
        pt_idx_in_zone = pt_idx_in_zone[attach_pts_mask]

        dist = np.linalg.norm((coords_in_zone - zone_centroid), axis=1)

        pt_closest_centroid = pt_idx_in_zone[np.argmin(dist)]

        zone_id_array[pt_closest_centroid, :] = izone
    

    baseline_attach1_zones_mat = (convert_idx_vec[baseline_attach1_zones_mat]).astype(np.int)    # Convert to the clustered indexing
    baseline_attach2_zones_mat = convert_idx_vec[baseline_attach2_zones_mat]

    # print(np.unique(baseline_attach1_zones_mat, return_counts=True))

    # import cv2
    # cv2.imwrite('test.jpg', baseline_attach1_zones_mat)
    # plot_sphere(phi_mat*np.pi/180, theta_mat*np.pi/180, baseline_attach1_zones_mat)
    # exit()

    zone_ids = np.arange(num_zones)+1
    num_theta_1D, num_phi_1D = baseline_attach1_zones_mat.shape
    N_tot_pts = baseline_attach1_zones_mat.size
    attachment_probabilities = np.zeros((num_zones))
    for row_idx, row in enumerate(baseline_attach1_zones_mat):
        theta = theta_mat[row_idx,0]*np.pi/180
        for pt in row:

            attachment_probabilities[pt-1] += np.sin(theta)*np.pi/num_theta_1D * 2*np.pi/num_phi_1D              # points are zero-indexed
    attachment_probabilities /= (4*np.pi)
    print(np.sum(attachment_probabilities))
    attachment_probabilities = {zone_ids[i]:attachment_probabilities[i] for i in np.arange(attachment_probabilities.size)}
    print('Attachment probabilities:', attachment_probabilities)
    # exit()

    num_occurrences = sum(count_arr[zone])
    attachment_probabilities[izone] = num_occurrences/tot_num_attach
    viz.visualize(solution['surf_mesh'], 2, {'scalars':{0: 'Attachment points'}}, 'attachment_points', True, zone_id_array, None, type='surface_mesh') # Can only have scalars on a surface mesh

    # Associate each point from the optimal attachment analysis with an attachment zone and reshape into array
    q_opt_pos_coords = surf_mesh['pcg'][q_opt_pos_attach_mat.ravel(),:]
    q_opt_neg_coords = surf_mesh['pcg'][q_opt_neg_attach_mat.ravel(),:]
    q_opt_pos_attach_zones_mat = convert_idx_vec[attachment_zones.predict(q_opt_pos_coords).reshape(q_opt_pos_attach_mat.shape)+1]   # Convert to 1-indexing
    q_opt_neg_attach_zones_mat = convert_idx_vec[attachment_zones.predict(q_opt_neg_coords).reshape(q_opt_neg_attach_mat.shape)+1]

    E_margin_optimal_charging = np.abs((q_opt_Eattach_mat-baseline_Eattach_mat)/baseline_Eattach_mat)

    # Temporary fix, not sure why the values are off by 180 degrees
    baseline_attach1_zones_mat = np.roll(baseline_attach1_zones_mat, baseline_attach1_zones_mat.shape[1]//2, axis=1)
    baseline_attach2_zones_mat = np.roll(baseline_attach2_zones_mat, baseline_attach2_zones_mat.shape[1]//2, axis=1)
    baseline_leader1_sign_mat = np.roll(baseline_leader1_sign_mat, baseline_leader1_sign_mat.shape[1]//2, axis=1)
    baseline_leader2_sign_mat = np.roll(baseline_leader2_sign_mat, baseline_leader2_sign_mat.shape[1]//2, axis=1)
    baseline_Eattach_mat = np.roll(baseline_Eattach_mat, baseline_Eattach_mat.shape[1]//2, axis=1)

    print('num_zones', num_zones)
    # Plot data summaries
    get_cmap_vals('tab20', len(zones_list))
    plot_summaries(theta_mat, 'theta', case)
    plot_summaries(phi_mat, 'phi', case)
    plot_summaries(baseline_attach1_zones_mat, 'attach_pt1', case, baseline_leader1_sign_mat, num_zones)
    plot_summaries(baseline_attach2_zones_mat, 'attach_pt2', case, baseline_leader2_sign_mat, num_zones)
    plot_summaries(baseline_leader1_sign_mat, 'leader1_sign', case)
    plot_summaries(baseline_leader2_sign_mat, 'leader2_sign', case)
    plot_summaries(baseline_Eattach_mat/1e3, 'baseline_Eattach', case)
    plot_summaries(q_opt_Eattach_mat/1e3, 'opt_Eattach', case)
    plot_summaries(q_opt_pos_attach_zones_mat, 'opt_pos_attach', case, None, num_zones)
    plot_summaries(q_opt_neg_attach_zones_mat, 'opt_neg_attach', case, None, num_zones)
    plot_summaries(q_opt_mat*1e3, 'q_opt', case, baseline_leader1_sign_mat)   # Convert to mC 
    plot_summaries(E_margin_optimal_charging, 'E_margin', case)

def plot_summaries(data, case, sim_case, sign_data=None, num_zones=None):
    plt.rcParams['text.usetex'] = True
    # To indicate which ones have a negative leader incepted first, try the cross-hatching here: https://stackoverflow.com/questions/14045709/selective-patterns-with-matplotlib-imshow (in crosshatch_test.py)
    figsize = (25, 15)
    title_fsize = 40
    label_fsize=40
    # showflag = False
    showflag = True
    
    if case == 'theta':
        cmap = 'turbo'
        __, ax1 = plt.subplots(figsize=figsize)
        im = ax1.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto')
        cbar = plt.colorbar(im, ax=ax1, aspect=50)
        cbar.ax.set_yticklabels(np.arange(10), fontsize=label_fsize)
        # cbar.ax.tick_params(labelsize=label_fsize)
        cbar.set_label('Theta', fontsize=label_fsize, rotation=90, labelpad=10)
        ax1.set_title('Theta', fontsize=title_fsize)

        xvalues = np.linspace(0, 360, num=13, endpoint=True).astype(np.int)
        yvalues = np.linspace(0, 180, num=7, endpoint=True).astype(np.int)
        ax1.set_xlabel(r'$\phi$ [degrees]', fontsize=label_fsize)
        ax1.set_ylabel(r'$\theta$ [degrees]', fontsize=label_fsize)
        ax1.set_xticks(xvalues)
        ax1.set_yticks(yvalues)
        ax1.set_xticklabels(xvalues, fontsize=label_fsize)
        ax1.set_yticklabels(yvalues, fontsize=label_fsize)

    elif case == 'phi':
        cmap = 'turbo'
        __, ax2 = plt.subplots(figsize=figsize)
        im = ax2.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto')
        cbar = plt.colorbar(im, ax=ax2, aspect=50)
        cbar.ax.tick_params(labelsize=label_fsize)
        # cbar.ax.set_yticklabels(np.arange(10), fontsize=label_fsize)
        # cbar.ax.tick_params(labelsize=label_fsize)
        cbar.set_label('Phi', fontsize=label_fsize, rotation=90, labelpad=10)
        ax2.set_title('Phi', fontsize=title_fsize)

        xvalues = np.linspace(0, 360, num=13, endpoint=True).astype(np.int)
        yvalues = np.linspace(0, 180, num=7, endpoint=True).astype(np.int)
        ax2.set_xlabel(r'$\phi$ [degrees]', fontsize=label_fsize)
        ax2.set_ylabel(r'$\theta$ [degrees]', fontsize=label_fsize)
        ax2.set_xticks(xvalues)
        ax2.set_yticks(yvalues)
        ax2.set_xticklabels(xvalues, fontsize=label_fsize)
        ax2.set_yticklabels(yvalues, fontsize=label_fsize)

    elif case == 'attach_pt1':

        __, ax3 = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap('tab20', num_zones)
        im = ax3.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto', vmin=np.min(data) - 0.5, vmax=num_zones + 0.5)
        cbar = plt.colorbar(im, ax=ax3, aspect=50)
        cbar.ax.tick_params(labelsize=label_fsize)

        cbar.set_label('Attachment Point 1', fontsize=label_fsize, rotation=90, labelpad=10)
        ax3.set_title('Attachment Point 1', fontsize=title_fsize)

        xvalues = np.linspace(0, 360, num=13, endpoint=True).astype(np.int)
        yvalues = np.linspace(0, 180, num=7, endpoint=True).astype(np.int)
        ax3.set_xlabel(r'$\phi$ [degrees]', fontsize=label_fsize)
        ax3.set_ylabel(r'$\theta$ [degrees]', fontsize=label_fsize)
        ax3.set_xticks(xvalues)
        ax3.set_yticks(yvalues)
        ax3.set_xticklabels(xvalues, fontsize=label_fsize)
        ax3.set_yticklabels(yvalues, fontsize=label_fsize)

        dx = 360/data.shape[1]
        dy = 180/data.shape[0]
        for idx, sign in enumerate(sign_data.ravel()):
            theta_idx = idx//data.shape[1]      # Row index, 'i' in output array
            phi_idx = idx%data.shape[1]      # Column index, 'j' in output array
            if sign < 0: # First leader was negative
                ax3.add_patch(matplotlib.patches.Rectangle((phi_idx*dx, theta_idx*dy), dx, dy, hatch='xx', fill=False, snap=False, linewidth=0))
        plt.grid()


    elif case == 'attach_pt2':
        # cmap='tab20'
        __, ax4 = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap('tab20', num_zones)
        im = ax4.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto', vmin=np.min(data) - 0.5, vmax=num_zones + 0.5)
        cbar = plt.colorbar(im, ax=ax4, aspect=50)
        cbar.ax.tick_params(labelsize=label_fsize)

        cbar.set_label('Attachment Point 2', fontsize=label_fsize, rotation=90, labelpad=10)
        ax4.set_title('Attachment Point 2', fontsize=title_fsize)

        xvalues = np.linspace(0, 360, num=13, endpoint=True).astype(np.int)
        yvalues = np.linspace(0, 180, num=7, endpoint=True).astype(np.int)
        ax4.set_xlabel(r'$\phi$ [degrees]', fontsize=label_fsize)
        ax4.set_ylabel(r'$\theta$ [degrees]', fontsize=label_fsize)
        ax4.set_xticks(xvalues)
        ax4.set_yticks(yvalues)
        ax4.set_xticklabels(xvalues, fontsize=label_fsize)
        ax4.set_yticklabels(yvalues, fontsize=label_fsize)

    elif case == 'leader1_sign':
        # cmap='turbo'
        __, ax5 = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap('tab20', np.max(data) - np.min(data) + 1)
        im = ax5.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto', vmin=np.min(data) - 0.5, vmax=np.max(data) + 0.5)
        # im = ax5.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto')
        cbar = plt.colorbar(im, ax=ax5, aspect=50, ticks=[-1, 1])
        cbar.ax.tick_params(labelsize=label_fsize)
        cbar.set_label('Leader 1 Sign', fontsize=label_fsize, rotation=90, labelpad=10)
        ax5.set_title('Leader 1 Sign', fontsize=title_fsize)

        xvalues = np.linspace(0, 360, num=13, endpoint=True).astype(np.int)
        yvalues = np.linspace(0, 180, num=7, endpoint=True).astype(np.int)
        ax5.set_xlabel(r'$\phi$ [degrees]', fontsize=label_fsize)
        ax5.set_ylabel(r'$\theta$ [degrees]', fontsize=label_fsize)
        ax5.set_xticks(xvalues)
        ax5.set_yticks(yvalues)
        ax5.set_xticklabels(xvalues, fontsize=label_fsize)
        ax5.set_yticklabels(yvalues, fontsize=label_fsize)

    elif case == 'leader2_sign':
        cmap='turbo'
        __, ax6 = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap('tab20', np.max(data) - np.min(data) + 1)
        im = ax6.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto', vmin=np.min(data) - 0.5, vmax=np.max(data) + 0.5)
        # im = ax6.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto')
        cbar = plt.colorbar(im, ax=ax6, aspect=50, ticks=[-1, 1])
        cbar.ax.tick_params(labelsize=label_fsize)

        cbar.set_label('Leader 2 Sign', fontsize=label_fsize, rotation=90, labelpad=10)
        ax6.set_title('Leader 2 Sign', fontsize=title_fsize)

        xvalues = np.linspace(0, 360, num=13, endpoint=True).astype(np.int)
        yvalues = np.linspace(0, 180, num=7, endpoint=True).astype(np.int)
        ax6.set_xlabel(r'$\phi$ [degrees]', fontsize=label_fsize)
        ax6.set_ylabel(r'$\theta$ [degrees]', fontsize=label_fsize)
        ax6.set_xticks(xvalues)
        ax6.set_yticks(yvalues)
        ax6.set_xticklabels(xvalues, fontsize=label_fsize)
        ax6.set_yticklabels(yvalues, fontsize=label_fsize)

    elif case == 'baseline_Eattach':
        cmap = 'turbo'
        __, ax7 = plt.subplots(figsize=figsize)
        im = ax7.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto', interpolation='bicubic')
        cbar = plt.colorbar(im, ax=ax7, aspect=50)
        
        ticklabels=np.concatenate(([round(np.min(data))], cbar.get_ticks(), [round(np.max(data))]))
        cbar.ax.set_yticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=label_fsize)
        
        cbar.set_label(r'Electric Field, $kV$', fontsize=label_fsize, rotation=90, labelpad=10)
        ax7.set_title(r'Electric Field for $Q_{ac}=0$', fontsize=title_fsize)

        xvalues = np.linspace(0, 360, num=13, endpoint=True).astype(np.int)
        yvalues = np.linspace(0, 180, num=7, endpoint=True).astype(np.int)
        ax7.set_xlabel(r'$\phi$ [degrees]', fontsize=label_fsize)
        ax7.set_ylabel(r'$\theta$ [degrees]', fontsize=label_fsize)
        ax7.set_xticks(xvalues)
        ax7.set_yticks(yvalues)
        ax7.set_xticklabels(xvalues, fontsize=label_fsize)
        ax7.set_yticklabels(yvalues, fontsize=label_fsize)

    elif case == 'opt_Eattach':
        cmap = 'turbo'
        __, ax8 = plt.subplots(figsize=figsize)
        im = ax8.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto', interpolation='bicubic')
        cbar = plt.colorbar(im, ax=ax8, aspect=50)
        cbar.set_label(r'Electric Field, $kV$', fontsize=label_fsize, rotation=90, labelpad=10)
        ticklabels=np.concatenate(([round(np.min(data))], cbar.get_ticks(), [round(np.max(data))]))
        cbar.ax.set_yticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=label_fsize)

        ax8.set_title('Electric Field for Optimal Charging Strategy', fontsize=title_fsize)

        xvalues = np.linspace(0, 360, num=13, endpoint=True).astype(np.int)
        yvalues = np.linspace(0, 180, num=7, endpoint=True).astype(np.int)
        ax8.set_xlabel(r'$\phi$ [degrees]', fontsize=label_fsize)
        ax8.set_ylabel(r'$\theta$ [degrees]', fontsize=label_fsize)
        ax8.set_xticks(xvalues)
        ax8.set_yticks(yvalues)
        ax8.set_xticklabels(xvalues, fontsize=label_fsize)
        ax8.set_yticklabels(yvalues, fontsize=label_fsize)

    elif case == 'opt_pos_attach':
        # cmap='tab20'
        __, ax9 = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap('tab20', num_zones)
        print(num_zones)
        im = ax9.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto', vmin=np.min(data) - 0.5, vmax=num_zones + 0.5)
        cbar = plt.colorbar(im, ax=ax9, aspect=50)
        cbar.ax.tick_params(labelsize=label_fsize)
        
        cbar.set_label('Positive Attachment Point', fontsize=label_fsize, rotation=90, labelpad=10)
        ax9.set_title('Positive Attachment Point - Optimum Charging Strategy', fontsize=title_fsize)

        xvalues = np.linspace(0, 360, num=13, endpoint=True).astype(np.int)
        yvalues = np.linspace(0, 180, num=7, endpoint=True).astype(np.int)
        ax9.set_xlabel(r'$\phi$ [degrees]', fontsize=label_fsize)
        ax9.set_ylabel(r'$\theta$ [degrees]', fontsize=label_fsize)
        ax9.set_xticks(xvalues)
        ax9.set_yticks(yvalues)
        ax9.set_xticklabels(xvalues, fontsize=label_fsize)
        ax9.set_yticklabels(yvalues, fontsize=label_fsize)

    elif case == 'opt_neg_attach':
        # cmap='tab20'
        __, ax10 = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap('tab20', num_zones)
        im = ax10.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto', vmin=np.min(data) - 0.5, vmax=num_zones + 0.5)
        cbar = plt.colorbar(im, ax=ax10, aspect=50)
        cbar.ax.tick_params(labelsize=label_fsize)
        
        cbar.set_label('Negative Attachment Point', fontsize=label_fsize, rotation=90, labelpad=10)
        ax10.set_title('Negative Attachment Point - Optimum Charging Strategy', fontsize=title_fsize)

        xvalues = np.linspace(0, 360, num=13, endpoint=True).astype(np.int)
        yvalues = np.linspace(0, 180, num=7, endpoint=True).astype(np.int)
        ax10.set_xlabel(r'$\phi$ [degrees]', fontsize=label_fsize)
        ax10.set_ylabel(r'$\theta$ [degrees]', fontsize=label_fsize)
        ax10.set_xticks(xvalues)
        ax10.set_yticks(yvalues)
        ax10.set_xticklabels(xvalues, fontsize=label_fsize)
        ax10.set_yticklabels(yvalues, fontsize=label_fsize)

    elif case == 'q_opt':
        min_val = np.min(data)
        max_val = np.max(data)
        neg_floor = -(-min_val-(-min_val)%.1)
        max_floor = max_val-max_val%.1
        ticks=np.linspace(neg_floor, max_floor, num=10)
        ticks = np.insert(ticks, 0, min_val)
        ticks = np.insert(ticks, -1, max_val)
        __, ax11 = plt.subplots(figsize=figsize)
        divnorm=colors.TwoSlopeNorm(vmin=np.min(data), vcenter=0., vmax=np.max(data))

        im = ax11.imshow(data, origin='lower', cmap=plt.get_cmap('RdBu_r'), extent=[0, 360, 0, 180], aspect='auto', interpolation='bicubic', norm=divnorm)
        cbar = plt.colorbar(im, ax=ax11, aspect=50, ticks=ticks)
        cbar.ax.set_yticklabels(['{:.2f}'.format(x) for x in ticks], fontsize=label_fsize)
        cbar.ax.tick_params(labelsize=label_fsize)
        cbar.set_label(r'Optimum Charge, $mC$', fontsize=label_fsize, rotation=90, labelpad=10)
        ax11.set_title('Optimal Charging Strategy', fontsize=title_fsize)

        xvalues = np.linspace(0, 360, num=13, endpoint=True).astype(np.int)
        yvalues = np.linspace(0, 180, num=7, endpoint=True).astype(np.int)
        ax11.set_xlabel(r'$\phi$ [degrees]', fontsize=label_fsize)
        ax11.set_ylabel(r'$\theta$ [degrees]', fontsize=label_fsize)
        ax11.set_xticks(xvalues)
        ax11.set_yticks(yvalues)
        ax11.set_xticklabels(xvalues, fontsize=label_fsize)
        ax11.set_yticklabels(yvalues, fontsize=label_fsize)

    elif case == 'E_margin':
        cmap = 'turbo'
        __, ax12 = plt.subplots(figsize=figsize)
        im = ax12.imshow(data, origin='lower', cmap=plt.get_cmap(cmap), extent=[0, 360, 0, 180], aspect='auto', interpolation='bicubic')
        cbar = plt.colorbar(im, ax=ax12, aspect=50)
        # cbar.ax.set_yticklabels(np.arange(10), fontsize=label_fsize)
        cbar.ax.tick_params(labelsize=label_fsize)
        cbar.set_label('Electric Field Margin', fontsize=label_fsize, rotation=90, labelpad=10)
        ax12.set_title('Electric Field Margin for Optimal Charging Strategy', fontsize=title_fsize)

        xvalues = np.linspace(0, 360, num=13, endpoint=True).astype(np.int)
        yvalues = np.linspace(0, 180, num=7, endpoint=True).astype(np.int)
        ax12.set_xlabel(r'$\phi$ [degrees]', fontsize=label_fsize)
        ax12.set_ylabel(r'$\theta$ [degrees]', fontsize=label_fsize)
        ax12.set_xticks(xvalues)
        ax12.set_yticks(yvalues)
        ax12.set_xticklabels(xvalues, fontsize=label_fsize)
        ax12.set_yticklabels(yvalues, fontsize=label_fsize)

    plt.tight_layout()
    if showflag:
        plt.show()
    else:
        plt.savefig('/media/homehd/saustin/lightning_research/3D-CG/postprocessing/local_analysis_directory/{}/analysis_output/{}.png'.format(sim_case, case))


    # ax = plt.subplots()
    # ax.imshow(baseline_first_attach_zones, interpolation='bicubic', extent=[0, 360, 0, 360])
    # for angle_idx, leader_sign_pair in enumerate(baseline_leader_sign_vec):
    #     theta_idx = angle_idx%num_1d_angle_pts      # Row index, 'i' in output array
    #     phi_idx = angle_idx//num_1d_angle_pts      # Column index, 'j' in output array
    #     if leader_sign_pair[0] == -1: # First leader was negative
    #         ax.add_patch(mpl.patches.Rectangle((theta_idx-.5, phi_idx-.5), 1, 1, hatch='///////', fill=False, snap=False))

    # plt.show()

    # plt.imshow(baseline_second_attach_zones, interpolation='bicubic', extent=[0, 360, 0, 360])

    # # DON'T forget to mirror the endpoints for both dimensions
    # # Q optimum case
    # plt.imshow(E_margin_optimal_charging, interpolation='bicubic', extent=[0, 360, 0, 360])
    # plt.imshow(q_opt_mat, interpolation='bicubic', extent=[0, 360, 0, 360])
    # plt.imshow(q_opt_pos_attach_zones, interpolation='bicubic', extent=[0, 360, 0, 360])
    # plt.imshow(q_opt_neg_attach_zones, interpolation='bicubic', extent=[0, 360, 0, 360])



    # plt.imshow(E_margin_optimal_charging, origin='lower', cmap=plt.get_cmap('turbo'), extent=[0, 180, 0, 360], aspect='auto')
    # plt.colorbar()
    # plt.title('Electric Field Margin Using Optimal Charging Strategy')
    # plt.xlabel(r'$\phi$, [degrees]')
    # plt.ylabel(r'$\theta$, [degrees]')
    # plt.show()
    # # plt.savefig(summaries_aggregate_dname+'E_margin_opt.png')

    # plt.imshow(q_opt_mat,origin="lower", cmap=plt.get_cmap('turbo'))
    # plt.colorbar()
    # plt.savefig(summaries_aggregate_dname+'q_opt.png')

    # plt.imshow(q_opt_pos_attach_zones_mat,origin="lower", cmap=plt.get_cmap('turbo'))
    # plt.colorbar()
    # plt.savefig(summaries_aggregate_dname+'q_opt_pos_attach_zones.png')

    # plt.imshow(q_opt_neg_attach_zones_mat,origin="lower", cmap=plt.get_cmap('turbo'))
    # plt.colorbar()
    # plt.savefig(summaries_aggregate_dname+'q_opt_neg_attach_zones.png')

    # plt.imshow(baseline_attach1_zones_mat, origin='lower', cmap=plt.get_cmap('turbo'), extent=[0, 180, 0, 360], aspect='auto')
    # plt.colorbar()
    # plt.title('Attachment 1 Zoning')
    # plt.xlabel(r'$\phi$, [degrees]')
    # plt.ylabel(r'$\theta$, [degrees]')
    # plt.fig
    # plt.show()


    # exit()
    # plt.imshow(baseline_attach2_zones_mat,origin="lower", cmap=plt.get_cmap('turbo'))
    # plt.colorbar()
    # plt.savefig(summaries_aggregate_dname+'baseline_attach2_zones.png')


if __name__ == '__main__':

    # ex: 'python local_attachment_postprocessing.py d8 120x240'
    case = sys.argv[1]
    grid_res = sys.argv[2]

    base_dir = './local_analysis_directory/'
    case_dir = '{}{}/'.format(base_dir, case)
    aggregated_data_dir = '{}aggregated_data_{}/'.format(case_dir, grid_res)
    output_data_dir = '{}analysis_output/'.format(case_dir)
    sol_fname = '{}{}_electrostatic_solution'.format(aggregated_data_dir, case)

    [os.makedirs(dir) for dir in [base_dir, case_dir, aggregated_data_dir, output_data_dir] if not os.path.exists(dir)]

    # os.system('scp saustin@txe1-login.mit.edu:/home/gridsan/saustin/research/3D-CG/postprocessing/fem_solutions/{}/attachment_analysis_{}/aggregated_data/* {}'.format(case, grid_res, aggregated_data_dir))
    
    # exit()
    local_attachment_postprocessing(aggregated_data_dir, sol_fname, case)
