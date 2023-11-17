import unittest
import numpy as np
from .linear_algebra import calc_jump_matrix, possible_message_states
from .simulation import weighted_mode 

class TestCalcJumpMatrix(unittest.TestCase):
    def test_uniform_jump_probabilities(self):
        Q1 = np.array([[-1., 1.],
                       [2., -2.]])
        J1 = calc_jump_matrix(Q1)
        expected_J1 = np.array([[0., 1.],
                                [1., 0.]])
        if not np.allclose(J1, expected_J1):
            print("Test 1")
            print("Calculated J1:\n", J1)
            print("Expected J1:\n", expected_J1)
        self.assertTrue(np.allclose(J1, expected_J1))

    def test_mixed_rates(self):
        Q2 = np.array([[-3., 1., 2.],
                       [3., -3., 0.],
                       [1., 2., -3.]])
        J2 = calc_jump_matrix(Q2)
        expected_J2 = np.array([[0., 1/3, 2/3],
                                [1, 0., 0.],
                                [1/3, 2/3, 0.]])
        if not np.allclose(J2, expected_J2):
            print("Test 2")
            print("Calculated J2:\n", J2)
            print("Expected J2:\n", expected_J2)
        self.assertTrue(np.allclose(J2, expected_J2))
    
    def test_possible_message_states(self):
        # Test for n=3
        matrix_3 = possible_message_states(3)
        expected_3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.assertEqual(matrix_3, expected_3)

        # Test for n=2
        matrix_2 = possible_message_states(2)
        expected_2 = [[1.0, 0.0], [0.0, 1.0]]
        self.assertEqual(matrix_2, expected_2)

        # Test for n=1
        matrix_1 = possible_message_states(1)
        expected_1 = [[1.0]]
        self.assertEqual(matrix_1, expected_1)

class TestWeightedStatistics(unittest.TestCase):
    
    def test_weighted_mode(self):
        values = [1., 2., 2., 3., 4.]
        weights = [1., 2., 1., 1., 1.]
        expected_mode = 2
        calculated_mode = weighted_mode(values, weights)
        self.assertEqual(calculated_mode, expected_mode)
    
    # Additional tests for weighted_mode
    def test_weighted_mode_multiple_modes(self):
        values = [1., 2., 2., 3., 3.]
        weights = [1., 2., 1., 2., 1.]
        expected_mode = 2.5  # Both 2 and 3 have the same highest weight
        calculated_mode = weighted_mode(values, weights)
        self.assertEqual(calculated_mode, expected_mode)

    # Additional tests for weighted_mode
    def test_weighted_mode_multiple_modes2(self):
        values =  [1., 2., 2., 3., 3., 2.1 ]
        weights = [1., 2., 1., 2., 1., 2.0]
        expected_mode = 2.1  
        calculated_mode = weighted_mode(values, weights)
        self.assertEqual(calculated_mode, expected_mode)

    def test_weighted_mode_empty_lists(self):
        values = []
        weights = []
        with self.assertRaises(ValueError):
            weighted_mode(values, weights)

    # Additional tests can be added here to test other edge cases or different scenarios

# This allows the test to be run from the command line
if __name__ == '__main__':
    unittest.main()