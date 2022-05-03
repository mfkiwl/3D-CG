import pyvista as pv
import numpy as np
import pickle
import os
import sys

def aggregate_sol(case_parent_dir, case_name, outdir):
    os.mkdir(outdir + '/' + case_name)

    print('Case:', case_name)
    print('Reading VTUs')
    phi_sol = pv.read(case_parent_dir + '/' + case_name +'/'+case_name +'_Phi.vtu')
    phi_sol_surf = pv.read(case_parent_dir + '/'  + case_name +'/'+case_name +'_Phi_surface.vtu')

    Ex_sol = pv.read(case_parent_dir + '/'  + case_name +'/'+case_name +'_Ex.vtu')
    Ex_sol_surf = pv.read(case_parent_dir + '/'  + case_name +'/'+case_name +'_Ex_surface.vtu')

    Ey_sol = pv.read(case_parent_dir + '/'  + case_name +'/'+case_name +'_Ey.vtu')
    Ey_sol_surf = pv.read(case_parent_dir + '/'  + case_name +'/'+case_name +'_Ey_surface.vtu')

    Ez_sol = pv.read(case_parent_dir + '/'  + case_name +'/'+case_name +'_Ez.vtu')
    Ez_sol_surf = pv.read(case_parent_dir + '/'  + case_name +'/'+case_name +'_Ez_surface.vtu')
 
    print('Writing volume solution')
    phi_sol.point_data['Ex_Potential'] = Ex_sol.point_data['Potential']
    phi_sol.point_data['Ex_Potential_Gradient'] = Ex_sol.point_data['Potential Gradient']
    phi_sol.point_data['Ey_Potential'] = Ey_sol.point_data['Potential']
    phi_sol.point_data['Ey_Potential_Gradient'] = Ey_sol.point_data['Potential Gradient']
    phi_sol.point_data['Ez_Potential'] = Ez_sol.point_data['Potential']
    phi_sol.point_data['Ez_Potential_Gradient'] = Ez_sol.point_data['Potential Gradient']
    phi_sol.save(outdir+'/' + case_name + '/' + case_name + '.vtu')

    print('Writing surface solution')
    phi_sol_surf.point_data['Ex_Surface Normal Electric Field'] = Ex_sol_surf.point_data['Surface Normal Electric Field']
    phi_sol_surf.point_data['Ey_Surface Normal Electric Field'] = Ey_sol_surf.point_data['Surface Normal Electric Field']
    phi_sol_surf.point_data['Ez_Surface Normal Electric Field'] = Ez_sol_surf.point_data['Surface Normal Electric Field']
    phi_sol_surf.save(outdir+'/' + case_name + '/' + case_name + '_surface.vtu')

    # Read in meshes and master data structure
    print('Reading meshes and master data structure')
    with open(case_parent_dir + '/'  + case_name +'/mesh_Phi', 'rb') as file:    # Use the Phi mesh as they're all the same
        vol_mesh = pickle.load(file)

    with open(case_parent_dir + '/'  + case_name +'/'+case_name + '_Phisurface_mesh', 'rb') as file:    # Use the Phi mesh as they're all the same
        surf_mesh = pickle.load(file)

    with open(case_parent_dir + '/'  + case_name +'/master', 'rb') as file:
        master = pickle.load(file)

    # Read in solution array
    print('Reading volume solution')
    with open(case_parent_dir + '/'  + case_name +'/'+case_name + '_Phi_surface_scalars.npy', 'rb') as file:
        Phi_surf = np.load(file)

    with open(case_parent_dir + '/'  + case_name +'/'+case_name + '_Ex_surface_scalars.npy', 'rb') as file:
        Ex_surf = np.load(file)

    with open(case_parent_dir + '/'  + case_name +'/'+case_name + '_Ey_surface_scalars.npy', 'rb') as file:
        Ey_surf = np.load(file)

    with open(case_parent_dir + '/'  + case_name +'/'+case_name + '_Ez_surface_scalars.npy', 'rb') as file:
        Ez_surf = np.load(file)

    # Read in surface solution
    print('Reading surface solution')
    with open(case_parent_dir + '/'  + case_name +'/'+case_name + '_Phi_solution.npy', 'rb') as file:
        Phi_vol = np.load(file)

    with open(case_parent_dir + '/'  + case_name +'/'+case_name + '_Ex_solution.npy', 'rb') as file:
        Ex_vol = np.load(file)

    with open(case_parent_dir + '/'  + case_name +'/'+case_name + '_Ey_solution.npy', 'rb') as file:
        Ey_vol = np.load(file)

    with open(case_parent_dir + '/'  + case_name +'/'+case_name + '_Ez_solution.npy', 'rb') as file:
        Ez_vol = np.load(file)

    # Load data structures
    fields = {}
    fields['vol_mesh'] = vol_mesh
    fields['surf_mesh'] = surf_mesh
    fields['master'] = master
    
    fields['Phi_pot_surf'] = Phi_surf[:,0]  # 1D vector
    fields['Ex_pot_surf'] = Ex_surf[:,0]  # 1D vector
    fields['Ey_pot_surf'] = Ey_surf[:,0]  # 1D vector
    fields['Ez_pot_surf'] = Ez_surf[:,0]  # 1D vector

    fields['Phi_grad_normal_surf'] = Phi_surf[:,1]  # 1D vector
    fields['Ex_grad_normal_surf'] = Ex_surf[:,1]  # 1D vector
    fields['Ey_grad_normal_surf'] = Ey_surf[:,1]  # 1D vector
    fields['Ez_grad_normal_surf'] = Ez_surf[:,1]  # 1D vector 


    fields['Phi_pot_vol'] = Phi_vol[:,0]  # 1D vector
    fields['Ex_pot_vol'] = Ex_vol[:,0]  # 1D vector
    fields['Ey_pot_vol'] = Ey_vol[:,0]  # 1D vector
    fields['Ez_pot_vol'] = Ez_vol[:,0]  # 1D vector

    fields['Phi_grad_vol'] = Phi_vol[:,1:]  # (npcg x 3) array
    fields['Ex_grad_vol'] = Ex_vol[:,1:]  # (npcg x 3) array
    fields['Ey_grad_vol'] = Ey_vol[:,1:]  # (npcg x 3) array
    fields['Ez_grad_vol'] = Ez_vol[:,1:]  # (npcg x 3) array
    
    with open(outdir+'/' + case_name + '/' + case_name + '_electrostatic_solution', 'w+b') as file:
        pickle.dump(fields, file)

    print()

if __name__ == '__main__':
    case_parent_dir = sys.argv[1]
    case_name = sys.argv[2]
    outdir = sys.argv[3]

    aggregate_sol(case_parent_dir, case_name, outdir)