import numpy as np

def get_E(fields, alpha, phi, units, type):
    # Note: this is assuming that the spatial coordinate is already dimensional - i.e. the Qac/C term is not divided by the fuselage radius Rf.
    if units == 'degrees':
        alpha *= np.pi/180
        phi *= np.pi/180

    if type == 'vol':
        E_Q = fields['Phi_grad_vol']
        E_x_vol = np.sin(alpha)*np.cos(phi)*-fields['Ex_grad_vol']
        E_y_vol = np.sin(alpha)*np.sin(phi)*-fields['Ey_grad_vol']
        E_z_vol = np.cos(alpha)*fields['Ez_grad_vol']
        E = E_x_vol + E_y_vol + E_z_vol
 
    elif type == 'surf':
        E_Q = fields['Phi_grad_normal_surf']
        E_x_surf = np.sin(alpha)*np.cos(phi)*-fields['Ex_grad_normal_surf'] # Negative sign to account for difference between CSYS in this research and Guerra 2018
        E_y_surf = np.sin(alpha)*np.sin(phi)*-fields['Ey_grad_normal_surf']
        E_z_surf = np.cos(alpha)*fields['Ez_grad_normal_surf']
        E = E_x_surf + E_y_surf + E_z_surf

    return E_Q, E

if __name__ == '__main__':
    fields = {'E_Q':1,'Ex':1,'Ey':1,'Ez':1}

    # In degrees!
    alpha = 20
    phi = -10
    get_E(fields, alpha, phi, 'degrees')