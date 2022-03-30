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
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube24', 'cg'), 0.019172725886541175, places=10)
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh24_solver_amg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube24', 'amg'), 0.019172731204138982, places=10)
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
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube100', 'cg'), 0.00908719483965248, places=10)
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh100_solver_amg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube100', 'amg'), 0.009087201373723763, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_homogeneous_dirichlet_cube_porder3_mesh4591_solver_cg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube4591', 'cg'), 0.00014311246139742106, places=10)
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh4591_solver_amg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertAlmostEqual(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube4591', 'amg'), 0.0007345822001354474, places=10)
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
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube24', 'cg'), 9.618263246968795e-08, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_amg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube24', 'amg'), 3.9033551280098777e-07, places=10)
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
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube100', 'cg'), 3.620905547063735e-07, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_amg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube100', 'amg'), 2.24349044913863e-06, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_linear_cube_porder3_mesh4591_solver_cg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube4591', 'cg'), 6.0267761911037e-06, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh4591_solver_amg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertAlmostEqual(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube4591', 'amg'), 0.07752102670151978, places=10)
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
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube24', 'cg'), 1.0414178408524322e-09, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_amg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube24', 'amg'), 1.5148356236061034e-08, places=10)
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
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube100', 'cg'), 1.3854304370397585e-09, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_amg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube100', 'amg'), 0.00031411458360119937, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_linear_cube_porder3_mesh4591_solver_cg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube4591', 'cg'), 1.2027956408644513e-09, places=10)
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh4591_solver_amg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertAlmostEqual(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube4591', 'amg'), 0.5037930045775671, places=10)
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
        self.assertRaises(ValueError, cube_sine_neumann.main.test_3d_cube_sine_neumann, 3, 'cube24', 'cg')
        os.chdir('../')
        pass

    def test_3d_sine_cube_porder3_mesh24_solver_amg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube24', 'amg'), 0.10932475985404677, places=10)
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
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube100', 'cg'), 0.009627469761016012, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_amg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube100', 'amg'), 0.0096560620319009, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_sine_cube_porder3_mesh4591_solver_cg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube4591', 'cg'), 0.0001654806516502294, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh4591_solver_amg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertAlmostEqual(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube4591', 'amg'), 0.3732191700124986, places=10)
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
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube24', 'cg'), 0.048132397953759565, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh24_solver_amg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube24', 'amg'), 0.04813250432172467, places=10)
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
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube100', 'cg'), 0.00733266104389263, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_amg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube100', 'amg'), 0.007333087953788953, places=10)
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_sine_cube_porder3_mesh4591_solver_cg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube4591', 'cg'), 0.000169693142116345, places=10)
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh4591_solver_amg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertAlmostEqual(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube4591', 'amg'), 0.06511536603526791, places=10)
        os.chdir('../')

if __name__ == '__main__':
    unittest.main(verbosity=2)

