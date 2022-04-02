# Test suite for the 3D FEM solver
# TODO: Improve the directory changing strategy

import unittest
import os

class FEM2DTests(unittest.TestCase):
    def test_2d_square_equals_matlab_result(self):
        os.chdir('./square_2d')
        import square_2d.main
        self.assertTrue(square_2d.main.test_2d_square())
        os.chdir('../')

class FEM3DTests(unittest.TestCase):
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
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube24', 'cg'), 0.01917272673225523, places=10)
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh24_solver_amg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube24', 'amg'), 0.019172726399754425, places=10)
        os.chdir('../')

    # Mesh of 100 tets
    def test_3d_homogeneous_dirichlet_cube_porder3_mesh100_solver_direct(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube100', 'direct'), 0.009087195260201297, places=10)
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh100_solver_cg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube100', 'cg'), 0.00908719413783865, places=10)
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh100_solver_amg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube100', 'amg'), 0.009087195898815015, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_homogeneous_dirichlet_cube_porder3_mesh4591_solver_cg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube4591', 'cg'), 0.00014311178044668083, places=10)
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh4591_solver_amg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube4591', 'amg'), 0.00014311365057617653, places=10)
        os.chdir('../')

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
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube24', 'cg'), 1.8157378972594174e-07, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_amg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube24', 'amg'), 2.2296259727383472e-08, places=10)
        os.chdir('../')

    # Mesh of 100 tets
    def test_3d_linear_cube_porder3_mesh100_solver_direct_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube100', 'direct'), 5.662137425588298e-15, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_cg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube100', 'cg'), 2.654373962296397e-07, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_amg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube100', 'amg'), 2.1957569323882709e-07, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_linear_cube_porder3_mesh4591_solver_cg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube4591', 'cg'), 3.340236856219647e-06, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh4591_solver_amg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube4591', 'amg'), 3.95948403109081e-06, places=10)
        os.chdir('../')

    # Zero forcing function with Neumann BC
    # Mesh of 24 tets
    def test_3d_linear_cube_porder3_mesh24_solver_direct_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube24', 'direct'), 7.627232179174825e-14, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_cg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube24', 'cg'), 1.1038926439610464e-09, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_amg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube24', 'amg'), 1.3353449457298439e-09, places=10)
        os.chdir('../')

    # Mesh of 24 tets
    def test_3d_linear_cube_porder3_mesh100_solver_direct_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube100', 'direct'), 7.904787935331115e-14, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_cg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube100', 'cg'), 5.85582138334928e-10, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_amg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube100', 'amg'), 1.8555359471150723e-09, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_linear_cube_porder3_mesh4591_solver_cg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube4591', 'cg'), 5.111532974666488e-09, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh4591_solver_amg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube4591', 'amg'), 2.60547150521262e-09, places=10)
        os.chdir('../')

    # Sinusoidal forcing function with Neumann BC
    # Mesh of 24 tets
    def test_3d_sine_cube_porder3_mesh24_solver_direct_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube24', 'direct'), 0.10932349169753576, places=10)
        os.chdir('../')

    # NOTE: This test case did not converge for some reason
    def test_3d_sine_cube_porder3_mesh24_solver_cg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube24', 'cg'), 0.10932362562649503, places=10)
        os.chdir('../')
        pass

    def test_3d_sine_cube_porder3_mesh24_solver_amg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube24', 'amg'), 0.10932360960969444, places=10)
        os.chdir('../')

    # Mesh of 100 tets
    def test_3d_sine_cube_porder3_mesh100_solver_direct_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube100', 'direct'), 0.009626935301285573, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_cg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube100', 'cg'), 0.009626927844773618, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_amg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube100', 'amg'), 0.00962747846680665, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_sine_cube_porder3_mesh4591_solver_cg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube4591', 'cg'), 0.0001760139216239187, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh4591_solver_amg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube4591', 'amg'), 0.00018070625629817982, places=10)
        os.chdir('../')

    # Sinusoidal forcing function with Dirichlet BC
    # Mesh of 24 tets
    def test_3d_sine_cube_porder3_mesh24_solver_direct_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube24', 'direct'), 0.0481321872511663, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh24_solver_cg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube24', 'cg'), 0.04813213613091227, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh24_solver_amg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube24', 'amg'), 0.048132224297690374, places=10)
        os.chdir('../')

    # Mesh of 100 tets
    def test_3d_sine_cube_porder3_mesh100_solver_direct_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube100', 'direct'), 0.007332617986654966, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_cg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube100', 'cg'), 0.007332607101341426, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_amg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube100', 'amg'), 0.007332663981044041, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_sine_cube_porder3_mesh4591_solver_cg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube4591', 'cg'), 0.00016798028525166764, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh4591_solver_amg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube4591', 'amg'), 0.0001705728968711684, places=10)
        os.chdir('../')

if __name__ == '__main__':
    unittest.main(verbosity=2)

