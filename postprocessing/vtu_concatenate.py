import pyvista as pv

# phi_sol = pv.read('./430K_Phi_out/boeing_430K_Phi.vtu')
# Ex_sol = pv.read('./430K_Ex_out/boeing_430K_Ex.vtu')
# Ey_sol = pv.read('./430K_Ey_out/boeing_430K_Ey.vtu')
# Ez_sol = pv.read('./430K_Ez_out/out/boeing_430K_Ez.vtu')

# print('done reading')
# phi_sol.point_data['Ex_Potential'] = Ex_sol.point_data['Potential']
# phi_sol.point_data['Ex_Potential_Gradient'] = Ex_sol.point_data['Potential Gradient']
# phi_sol.point_data['Ey_Potential'] = Ey_sol.point_data['Potential']
# phi_sol.point_data['Ey_Potential_Gradient'] = Ey_sol.point_data['Potential Gradient']
# phi_sol.point_data['Ez_Potential'] = Ez_sol.point_data['Potential']
# phi_sol.point_data['Ez_Potential_Gradient'] = Ez_sol.point_data['Potential Gradient']

# phi_sol.save('430K_combined.vtu')

phi_sol = pv.read('./6M_Phi_out/boeing_6M_Phi.vtu')
Ex_sol = pv.read('./6M_Ex_out/boeing_6M_Ex.vtu')
Ey_sol = pv.read('./6M_Ey_out/boeing_6M_Ey.vtu')
Ez_sol = pv.read('./6M_Ez_out/boeing_6M_Ez.vtu')

print('done reading')
phi_sol.point_data['Ex_Potential'] = Ex_sol.point_data['Potential']
phi_sol.point_data['Ex_Potential_Gradient'] = Ex_sol.point_data['Potential Gradient']
phi_sol.point_data['Ey_Potential'] = Ey_sol.point_data['Potential']
phi_sol.point_data['Ey_Potential_Gradient'] = Ey_sol.point_data['Potential Gradient']
phi_sol.point_data['Ez_Potential'] = Ez_sol.point_data['Potential']
phi_sol.point_data['Ez_Potential_Gradient'] = Ez_sol.point_data['Potential Gradient']

phi_sol.save('6M_combined.vtu')