# Test suite for the 3D FEM solver

import unittest
import os

class QuadratureTests(unittest.TestCase):
    def test_single_tet_quadrature_porder1(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertAlmostEqual(single_tet_quadrature.single_tet_quadrature.single_tet(1), 0, places=10)
        os.chdir('../')

    def test_single_tet_quadrature_porder2(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertAlmostEqual(single_tet_quadrature.single_tet_quadrature.single_tet(2), 0, places=10)
        os.chdir('../')

    def test_single_tet_quadrature_porder3(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertAlmostEqual(single_tet_quadrature.single_tet_quadrature.single_tet(3), 0, places=10)
        os.chdir('../')

    def test_single_tet_quadrature_porder4(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertAlmostEqual(single_tet_quadrature.single_tet_quadrature.single_tet(4), 0, places=10)
        os.chdir('../')

class SquareSine2DTests(unittest.TestCase):
    def test_2d_square_equals_matlab_result(self):
        os.chdir('./square_2d')
        import square_2d.main
        self.assertTrue(square_2d.main.test_2d_square())
        os.chdir('../')

class HomogeneousDirichletSinusoidTests(unittest.TestCase):
    # Sinusoidal forcing function with homogeneous dirichlet BCs
    # Mesh of 24 tets
    def test_3d_homogeneous_dirichlet_cube_porder3_mesh24_solver_direct(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube24', 'direct'), 0.01917272588654184, places=10)
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh24_solver_cg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube24', 'cg'), 0.019172725897400156, places=10)
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh24_solver_gmres(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube24', 'gmres'), 0.019172725888850772, places=10)
        os.chdir('../')

    # Mesh of 100 tets
    def test_3d_homogeneous_dirichlet_cube_porder3_mesh100_solver_direct(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube100', 'direct'), 0.009087195260202297, places=10)
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh100_solver_cg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube100', 'cg'), 0.009087195273018711, places=10)
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh100_solver_gmres(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube100', 'gmres'), 0.009087195258473457, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_homogeneous_dirichlet_cube_porder3_mesh4591_solver_cg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube4591', 'cg'), 0.0001431125240785036, places=10)
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh4591_solver_gmres(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube4591', 'gmres'), 0.00014311251431076144, places=10)
        os.chdir('../')

class LinearDirichletTests(unittest.TestCase):
    # Zero forcing function with dirichlet BCs - linear solution
    # Mesh of 24 tets
    def test_3d_linear_cube_porder3_mesh24_solver_direct_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube24', 'direct'), 5.773159728050814e-15, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_cg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube24', 'cg'), 1.0474859868381259e-10, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_gmres_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube24', 'gmres'), 1.2847334307508618e-11, places=10)
        os.chdir('../')

    # Mesh of 100 tets
    def test_3d_linear_cube_porder3_mesh100_solver_direct_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube100', 'direct'), 8.104628079763643e-15, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_cg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube100', 'cg'), 3.3875940941285876e-10, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_gmres_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube100', 'gmres'), 1.576550001658461e-10, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_linear_cube_porder3_mesh4591_solver_cg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube4591', 'cg'), 4.414924925644215e-09, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh4591_solver_gmres_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube4591', 'gmres'), 1.432073998275385e-08, places=10)
        os.chdir('../')

class LinearNeumannTests(unittest.TestCase):
    # Zero forcing function with Neumann BC
    # Mesh of 24 tets
    def test_3d_linear_cube_porder3_mesh24_solver_direct_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube24', 'direct'), 6.750155989720952e-14, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_cg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube24', 'cg'), 3.3936409238322085e-11, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_gmres_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube24', 'gmres'), 1.659117287999834e-11, places=10)
        os.chdir('../')

    # Mesh of 24 tets
    def test_3d_linear_cube_porder3_mesh100_solver_direct_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube100', 'direct'), 8.737455203799982e-14, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_cg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube100', 'cg'), 6.915801264995025e-12, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_gmres_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube100', 'gmres'), 1.1160961044254236e-11, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_linear_cube_porder3_mesh4591_solver_cg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube4591', 'cg'), 1.2409295813142762e-11, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh4591_solver_gmres_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube4591', 'gmres'), 8.894096570344345e-11, places=10)
        os.chdir('../')

class NeumannModSineTests(unittest.TestCase):
    # Sinusoidal forcing function with Neumann BC
    # Mesh of 24 tets
    def test_3d_sine_cube_porder3_mesh24_solver_direct_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube24', 'direct'), 0.10932349169753643, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh24_solver_cg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube24', 'cg'), 0.10932349156814838, places=10)
        os.chdir('../')
        pass

    def test_3d_sine_cube_porder3_mesh24_solver_gmres_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube24', 'gmres'), 0.10932349169753643, places=10)
        os.chdir('../')

    # Mesh of 100 tets
    def test_3d_sine_cube_porder3_mesh100_solver_direct_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube100', 'direct'), 0.00962693530128276, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_cg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube100', 'cg'), 0.009626935502042145, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_gmres_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube100', 'gmres'), 0.009626935152736405, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_sine_cube_porder3_mesh4591_solver_cg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube4591', 'cg'), 0.00016785903196614438, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh4591_solver_gmres_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube4591', 'gmres'), 0.00016786696395776346, places=10)
        os.chdir('../')

class DirichletModSineTests(unittest.TestCase):
    # Sinusoidal forcing function with Dirichlet BC
    # Mesh of 24 tets
    def test_3d_sine_cube_porder3_mesh24_solver_direct_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube24', 'direct'), 0.04813218725116686, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh24_solver_cg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube24', 'cg'), 0.04813218731078095, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh24_solver_gmres_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube24', 'gmres'), 0.04813218726306512, places=10)
        os.chdir('../')

    # Mesh of 100 tets
    def test_3d_sine_cube_porder3_mesh100_solver_direct_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube100', 'direct'), 0.0073326179866539665, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_cg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube100', 'cg'), 0.007332618293576121, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_gmres_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube100', 'gmres'), 0.007332617988105472, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_sine_cube_porder3_mesh4591_solver_cg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube4591', 'cg'), 0.00016783038514711635, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh4591_solver_gmres_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube4591', 'gmres'), 0.00016782867764086884, places=10)
        os.chdir('../')

if __name__ == '__main__':
    unittest.main(verbosity=2)