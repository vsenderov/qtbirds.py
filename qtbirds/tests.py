import unittest
import numpy as np
from .linear_algebra import calc_jump_matrix, possible_message_states

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

    # Additional tests can be added here to test other edge cases or different scenarios

# This allows the test to be run from the command line
if __name__ == '__main__':
    unittest.main()