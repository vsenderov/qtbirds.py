import numpy as np

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

def possible_message_states(n):
    return [[1.0 if row == col else 0.0 for col in range(n)] for row in range(n)]


