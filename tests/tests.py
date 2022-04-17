# Test suite for the 3D FEM solver

import unittest
import os
import numpy as np

class SingleElemQuadratureTests(unittest.TestCase):
    def test_single_tet_quadrature_porder1(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertAlmostEqual(single_tet_quadrature.single_tet_quadrature.single_tet_quadrature(1), 0, places=12)
        os.chdir('../')

    def test_single_tet_quadrature_porder2(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertAlmostEqual(single_tet_quadrature.single_tet_quadrature.single_tet_quadrature(2), 0, places=12)
        os.chdir('../')

    def test_single_tet_quadrature_porder3(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertAlmostEqual(single_tet_quadrature.single_tet_quadrature.single_tet_quadrature(3), 0, places=12)
        os.chdir('../')

    def test_single_tet_quadrature_porder4(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertAlmostEqual(single_tet_quadrature.single_tet_quadrature.single_tet_quadrature(4), 0, places=12)
        os.chdir('../')

    def test_single_tet_gradient_porder1(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertTrue(single_tet_quadrature.single_tet_quadrature.single_tet_gradients(1))
        os.chdir('../')

    def test_single_tet_gradient_porder2(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertTrue(single_tet_quadrature.single_tet_quadrature.single_tet_gradients(2))
        os.chdir('../')

    def test_single_tet_gradient_porder3(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertTrue(single_tet_quadrature.single_tet_quadrature.single_tet_gradients(3))
        os.chdir('../')

    def test_single_tet_gradient_porder4(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertTrue(single_tet_quadrature.single_tet_quadrature.single_tet_gradients(4))
        os.chdir('../')

    def test_single_tet_normals_porder1(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertTrue(single_tet_quadrature.single_tet_quadrature.single_tet_normals(1))
        os.chdir('../')

    def test_single_tet_normals_porder2(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertTrue(single_tet_quadrature.single_tet_quadrature.single_tet_normals(2))
        os.chdir('../')

    def test_single_tet_normals_porder3(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertTrue(single_tet_quadrature.single_tet_quadrature.single_tet_normals(3))
        os.chdir('../')

    def test_single_tet_normals_porder4(self):
        os.chdir('./single_tet_quadrature')
        import single_tet_quadrature.single_tet_quadrature
        self.assertTrue(single_tet_quadrature.single_tet_quadrature.single_tet_normals(4))
        os.chdir('../')


class MultiElemQuadratureTests(unittest.TestCase):
    def test_cube_quadrature_vol_porder3(self):
        os.chdir('./multielement_quadrature_test')
        import multielement_quadrature_test.multielem_quad
        self.assertAlmostEqual(multielement_quadrature_test.multielem_quad.volume_unit_cube(), 1, places=12)
        os.chdir('../')

    def test_cube_quadrature_one_face_porder3(self):
        os.chdir('./multielement_quadrature_test')
        import multielement_quadrature_test.multielem_quad
        self.assertAlmostEqual(multielement_quadrature_test.multielem_quad.area_unit_square_one_face(), 1, places=12)
        os.chdir('../')

    def test_cube_quadrature_all_faces_porder3(self):
        os.chdir('./multielement_quadrature_test')
        import multielement_quadrature_test.multielem_quad
        self.assertAlmostEqual(multielement_quadrature_test.multielem_quad.unit_square_total_surf_area(), 6, places=12)
        os.chdir('../')

class SquareSine2DTests(unittest.TestCase):
    def test_2d_square_equals_matlab_result(self):
        os.chdir('./square_2d')
        import square_2d.main
        self.assertTrue(square_2d.main.test_2d_square())
        os.chdir('../')

class HomogeneousDirichletSinusoidTests(unittest.TestCase):
    # Sinusoidal forcing function with homogeneous dirichlet BCs

    # porder 2
    def test_3d_homogeneous_dirichlet_cube_porder2_mesh24_solver_direct(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(2, 'cube24', 'direct'), (0.032895590899645, 0.698598418287601), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder2_mesh100_solver_gmres(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(2, 'cube100', 'gmres'), (0.11865077356709586, 0.4733916872295998), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder2_mesh4591_solver_gmres(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(2, 'cube4591', 'gmres'), (0.001305322415915855, 0.10459851034414536), atol=1e-10, rtol=0))
        os.chdir('../')

    # porder 3

    # Mesh of 24 tets
    def test_3d_homogeneous_dirichlet_cube_porder3_mesh24_solver_direct(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube24', 'direct'), (0.019172725886541286, 0.8323533494344479), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh24_solver_cg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube24', 'cg'), (0.019172725897400156, 0.832353349595786), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh24_solver_gmres(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube24', 'gmres'), (0.019172725888850772, 0.8323533494074837), atol=1e-10, rtol=0))
        os.chdir('../')

    # Mesh of 100 tets
    def test_3d_homogeneous_dirichlet_cube_porder3_mesh100_solver_direct(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube100', 'direct'), (0.009087195260202297, 0.2061914091182886), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh100_solver_cg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube100', 'cg'), (0.009087195273018711, 0.20619140877303765), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh100_solver_gmres(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube100', 'gmres'), (0.009087195258473457, 0.20619140909511424), atol=1e-10, rtol=0))
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_homogeneous_dirichlet_cube_porder3_mesh4591_solver_cg(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube4591', 'cg'), (0.0001431125240785036, 0.006311862525429407), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_homogeneous_dirichlet_cube_porder3_mesh4591_solver_gmres(self):
        os.chdir('./cube_sine_homogeneous_dirichlet')
        import cube_sine_homogeneous_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_homogeneous_dirichlet.main.test_3d_cube_sine_homoegeneous_dirichlet(3, 'cube4591', 'gmres'), (0.00014311251431076144, 0.006311862603773543), atol=1e-10, rtol=0))
        os.chdir('../')

class LinearDirichletTests(unittest.TestCase):
    # Zero forcing function with dirichlet BCs - linear solution

    # porder 2
    def test_3d_linear_cube_porder2_mesh24_solver_direct_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertTrue(np.allclose(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(2, 'cube24', 'direct'), (7.771561172376096e-16, 5.10702591327572e-15), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder2_mesh100_solver_gmres_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertTrue(np.allclose(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(2, 'cube100', 'gmres'), (1.4884049548413714e-10, 1.2466720869083074e-09), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder2_mesh4591_solver_gmres_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertTrue(np.allclose(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(2, 'cube4591', 'gmres'), (6.9500670774047535e-09, 3.639464307703122e-08), atol=1e-10, rtol=0))
        os.chdir('../')

    # Mesh of 24 tets - porder 3
    def test_3d_linear_cube_porder3_mesh24_solver_direct_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertTrue(np.allclose(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube24', 'direct'), (5.773159728050814e-15, 9.303668946358812e-14), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_cg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertTrue(np.allclose(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube24', 'cg'), (1.0474859868381259e-10, 9.394354183456244e-10), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_gmres_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertTrue(np.allclose(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube24', 'gmres'), (1.2847334307508618e-11, 1.7062673496326397e-10), atol=1e-10, rtol=0))
        os.chdir('../')

    # Mesh of 100 tets
    def test_3d_linear_cube_porder3_mesh100_solver_direct_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertTrue(np.allclose(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube100', 'direct'), (8.104628079763643e-15, 1.0169642905566434e-13), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_cg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertTrue(np.allclose(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube100', 'cg'), (3.3875940941285876e-10, 1.2208278832304131e-09), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_gmres_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertTrue(np.allclose(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube100', 'gmres'), (1.576550001658461e-10, 1.863406096092035e-09), atol=1e-10, rtol=0))
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_linear_cube_porder3_mesh4591_solver_cg_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertTrue(np.allclose(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube4591', 'cg'), (4.414924925644215e-09, 6.104446381716144e-08), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh4591_solver_gmres_dirichlet(self):
        os.chdir('./cube_linear_dirichlet')
        import cube_linear_dirichlet.main
        self.assertTrue(np.allclose(cube_linear_dirichlet.main.test_3d_cube_linear_dirichlet(3, 'cube4591', 'gmres'), (1.432073998275385e-08, 8.550128927087286e-08), atol=1e-10, rtol=0))
        os.chdir('../')

class LinearNeumannTests(unittest.TestCase):
    # Zero forcing function with Neumann BC
 
    # porder 2
    def test_3d_linear_cube_porder2_mesh24_solver_direct_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertTrue(np.allclose(cube_linear_neumann.main.test_3d_cube_linear_neumann(2, 'cube24', 'direct'), (3.019806626980426e-14, 3.241851231905457e-14), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder2_mesh100_solver_gmres_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertTrue(np.allclose(cube_linear_neumann.main.test_3d_cube_linear_neumann(2, 'cube100', 'gmres'), (9.710343640279007e-12, 5.511264777879887e-10), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder2_mesh4591_solver_gmres_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertTrue(np.allclose(cube_linear_neumann.main.test_3d_cube_linear_neumann(2, 'cube4591', 'gmres'), (1.308442243441732e-11, 1.4691803329469622e-10), atol=1e-10, rtol=0))
        os.chdir('../')


    # porder 3
    # Mesh of 24 tets
    def test_3d_linear_cube_porder3_mesh24_solver_direct_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertTrue(np.allclose(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube24', 'direct'), (6.750155989720952e-14, 1.2112533198660458e-13), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_cg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertTrue(np.allclose(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube24', 'cg'), (3.393529901529746e-11, 9.0934149099553e-10), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh24_solver_gmres_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertTrue(np.allclose(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube24', 'gmres'), (1.6590839813090952e-11, 1.2617595857022934e-10), atol=1e-10, rtol=0))
        os.chdir('../')

    # Mesh of 24 tets
    def test_3d_linear_cube_porder3_mesh100_solver_direct_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertTrue(np.allclose(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube100', 'direct'), (8.704148513061227e-14, 1.603162047558726e-13), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_cg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertTrue(np.allclose(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube100', 'cg'), (6.916689443414725e-12, 5.937016434032216e-10), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh100_solver_gmres_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertTrue(np.allclose(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube100', 'gmres'), (1.1162182289581324e-11, 1.233302942932376e-10), atol=1e-10, rtol=0))
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_linear_cube_porder3_mesh4591_solver_cg_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertTrue(np.allclose(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube4591', 'cg'), (1.2408185590118137e-11, 2.020048572859423e-09), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_linear_cube_porder3_mesh4591_solver_gmres_neumann(self):
        os.chdir('./cube_linear_neumann')
        import cube_linear_neumann.main
        self.assertTrue(np.allclose(cube_linear_neumann.main.test_3d_cube_linear_neumann(3, 'cube4591', 'gmres'), (8.894007752502375e-11, 4.465235958761582e-10), atol=1e-10, rtol=0))
        os.chdir('../')

class NeumannModSineTests(unittest.TestCase):
    # Sinusoidal forcing function with Neumann BC

    # porder 2
    def test_3d_sine_cube_porder2_mesh24_solver_direct_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertTrue(np.allclose(cube_sine_neumann.main.test_3d_cube_sine_neumann(2, 'cube24', 'direct'), (0.10561609155605095, 3.2814445800361454), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder2_mesh100_solver_gmres_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertTrue(np.allclose(cube_sine_neumann.main.test_3d_cube_sine_neumann(2, 'cube100', 'gmres'), (0.037242433300674116, 1.1939459448935166), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder2_mesh4591_solver_gmres_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertTrue(np.allclose(cube_sine_neumann.main.test_3d_cube_sine_neumann(2, 'cube4591', 'gmres'), (0.0024155638882124362, 0.2011908299791081), atol=1e-10, rtol=0))
        os.chdir('../')

    # porder 3
    # Mesh of 24 tets
    def test_3d_sine_cube_porder3_mesh24_solver_direct_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertTrue(np.allclose(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube24', 'direct'), (0.10932349169753579, 1.904797645851108), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh24_solver_cg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertTrue(np.allclose(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube24', 'cg'), (0.10932349156814825, 1.9047976463645488), atol=1e-10, rtol=0))
        os.chdir('../')
        pass

    def test_3d_sine_cube_porder3_mesh24_solver_gmres_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertTrue(np.allclose(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube24', 'gmres'), (0.10932349166698398, 1.9047976458142974), atol=1e-10, rtol=0))
        os.chdir('../')

    # Mesh of 100 tets
    def test_3d_sine_cube_porder3_mesh100_solver_direct_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertTrue(np.allclose(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube100', 'direct'), (0.00962693530128277, 0.2878495571225912), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_cg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertTrue(np.allclose(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube100', 'cg'), (0.009626935502042666, 0.2878495568375401), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_gmres_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertTrue(np.allclose(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube100', 'gmres'), (0.00962693515273753, 0.2878495571198103), atol=1e-10, rtol=0))
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_sine_cube_porder3_mesh4591_solver_cg_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertTrue(np.allclose(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube4591', 'cg'), (0.0001678590319662554, 0.019028382916581228), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh4591_solver_gmres_neumann(self):
        os.chdir('./cube_sine_neumann')
        import cube_sine_neumann.main
        self.assertTrue(np.allclose(cube_sine_neumann.main.test_3d_cube_sine_neumann(3, 'cube4591', 'gmres'), (0.00016786696396464684, 0.019028376558360198), atol=1e-10, rtol=0))
        os.chdir('../')

class DirichletModSineTests(unittest.TestCase):
    # Sinusoidal forcing function with Dirichlet BC

    # porder 2
    def test_3d_sine_cube_porder2_mesh24_solver_direct_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(2, 'cube24', 'direct'), (0.10668667232674173, 3.3331706222589874), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder2_mesh100_solver_gmres_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(2, 'cube100', 'gmres'), (0.03749403933993156, 1.2067383386850974), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder2_mesh4591_solver_gmres_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(2, 'cube4591', 'gmres'), (0.0018624121609238653, 0.1882010210693359), atol=1e-10, rtol=0))
        os.chdir('../')

    # porder 3
    # Mesh of 24 tets
    def test_3d_sine_cube_porder3_mesh24_solver_direct_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube24', 'direct'), (0.04813218725116686, 1.5118541685710767), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh24_solver_cg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube24', 'cg'), (0.04813218731078095, 1.511854168861702), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh24_solver_gmres_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube24', 'gmres'), (0.04813218726306512, 1.5118541685535565), atol=1e-10, rtol=0))
        os.chdir('../')

    # Mesh of 100 tets
    def test_3d_sine_cube_porder3_mesh100_solver_direct_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube100', 'direct'), (0.0073326179866539665, 0.3144455292632866), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_cg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube100', 'cg'), (0.007332618293576121, 0.3144455274942164), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh100_solver_gmres_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube100', 'gmres'), (0.007332617988105472, 0.31444552932709957), atol=1e-10, rtol=0))
        os.chdir('../')

    # Mesh of 4591 tets
    def test_3d_sine_cube_porder3_mesh4591_solver_cg_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube4591', 'cg'), (0.00016783038514711635, 0.01902837018074194), atol=1e-10, rtol=0))
        os.chdir('../')

    def test_3d_sine_cube_porder3_mesh4591_solver_gmres_dirichlet(self):
        os.chdir('./cube_sine_dirichlet')
        import cube_sine_dirichlet.main
        self.assertTrue(np.allclose(cube_sine_dirichlet.main.test_3d_cube_sine_dirichlet(3, 'cube4591', 'gmres'), (0.00016782867764086884, 0.019028340879067414), atol=1e-10, rtol=0))
        os.chdir('../')

class MasternodesPermTest(unittest.TestCase):
    def test_masternodes_perm_porder3(self):
        import masternodes_test
        self.assertTrue(masternodes_test.masternodes_perm_test(3))

    def test_masternodes_perm_porder2(self):
        import masternodes_test
        self.assertTrue(masternodes_test.masternodes_perm_test(2))

if __name__ == '__main__':
    unittest.main(verbosity=2)