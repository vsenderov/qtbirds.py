import numpy as np
import unittest

def calc_jump_matrix(Q):
    """
    This function takes a rate matrix Q of a continuous-time Markov chain (CTMC)
    and outputs the embedded jump chain matrix J.
    
    The jump matrix J represents the probabilities of transitioning from one state
    to another given that a transition has occurred, excluding self-transitions.
    
    :param Q: A square numpy array representing the rate matrix of a CTMC.
    :return: A numpy array representing the jump matrix J.
    """
    # Initialize J with zeros, having the same dimensions as Q
    J = np.zeros_like(Q)
    
    # Iterate over rows and columns of Q to compute J
    for row in range(Q.shape[0]):
        for col in range(Q.shape[1]):
            if row != col:
                # Off-diagonal entries: Q[row][col] / -Q[row][row]
                J[row][col] = Q[row][col] / -Q[row][row]
            # Diagonal entries in J are set to 0 as self-transitions are excluded
                
    return J

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


# This allows the test to be run from the command line
if __name__ == '__main__':
    unittest.main()
