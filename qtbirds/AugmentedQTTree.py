from __future__ import annotations # to shut up Pylance about cls: AugmentedQTNode
from .QTTree import QTNode, QTLeaf
import json
from scipy.stats import poisson
import numpy as np
from typing import Optional, Dict, Tuple, List, Union
from .linear_algebra import calc_jump_matrix, possible_message_states
from scipy.linalg import expm
import random



def evolve_message(x:np.ndarray, t: float, mod:ModelDynamics, sjumps:int) -> np.ndarray:
    """
    Evolve a message `x` based on model dynamics `mod` and a given number of simultaneous jumps `sjumps`.

    Parameters:
        x (ndarray): The initial message to evolve.
        mod (ModelDynamics): The dynamics of the model, containing matrices `jMol` and `qMol` and rate `nu`.
        sjumps (int): The number of simultaneous jumps to consider in the evolution.
        
    Returns:
        ndarray: The evolved message of the same shape as `x`.

    Linear Algebra:
        - If x[0, 0] < 0, indicating missing data, the function returns `x` unchanged.
        - Computes the matrix exponential `expM = exp(mod.qMol * t)`, where `t` should be defined or obtained in context.
        - Calculates the matrix power `jPower = mod.jMol ^ sjumps` (element-wise matrix power).
        - The resulting message is computed as `x @ jPower @ expM`.
    """
    # Check for missing data
    if np.any(x) < 0:
        return x  # Return unchanged if the message indicates missing data
    
    # Compute the matr ix exponential of qMol * t
    expM = expm(mod.q_mol * t)

    # Calculate the matrix power of jMol raised to the 'sjumps'
    jPower = np.linalg.matrix_power(mod.j_mol, sjumps)

    # Perform the matrix multiplication
    res = x @ jPower @ expM

    return res


def evolve_char_message(x:np.ndarray, t: float, mod: ModelDynamics, sjumps: int)->np.ndarray:
    """
    no missing data allowed
    """
    expM = expm(mod.q_char * t)
    jPower = np.linalg.matrix_power(mod.j_char, sjumps)
    res = x @ jPower @ expM
    return res


def multiply_or_pass(msg1, msg2):
    """
    Multiply elements of two vectors element-wise unless there is missing data
    indicated by negative values in any vector.

    Parameters:
        msg1 (np.ndarray): First message vector.
        msg2 (np.ndarray): Second message vector.

    Returns:
        np.ndarray: Resulting vector after element-wise multiplication or original vector with missing data.
    """
    if np.any(msg1 < 0):
        return msg1
    elif np.any(msg2 < 0):
        return msg2
    else:
        return msg1 * msg2


def compute_message_log_likelihood(mes: np.ndarray) -> float:
    """
    TODO incorporate equilibrium frequencies
    """
    if np.any(mes) < 0:
        return 0.
    
    l = mes.shape[0]  # Adjusted to get the length of the 1D array
    # Create an array 'base' filled with 1.0, repeated 'l' times
    base = np.ones((l, 1))  # Directly creating as a column vector
    # Matrix multiplication of 'mes' reshaped as a row vector with 'base' as a column vector
    # Reshape 'mes' to be a 1xL matrix (row vector)
    msum = mes.reshape(1, l) @ base  # Result is a 1x1 matrix

    # Access the first element of the resulting matrix 'msum', assuming it's 1x1, to get 'msumReal'
    msumReal = msum[0, 0]  # This will always work since msum is 1x1

    # Return the logarithm of 'msumReal'
    return np.log(msumReal)


class ModelDynamics:
    q_char: np.ndarray
    j_char: np.ndarray
    q_mol: np.ndarray
    j_mol: np.ndarray
    char_messages: List[List[float]]  # Possible initial character messages (depends on the number of character states)
    # molMessages: List[List[float]]  # Uncomment if you need this field

    def __init__(self, lam: float, norm_q_char: np.ndarray, mu: float, norm_q_mol: np.ndarray):
        self.q_char = lam*norm_q_char
        self.j_char = calc_jump_matrix(norm_q_char) 
        self.q_mol = mu*norm_q_mol
        self.j_mol = calc_jump_matrix(norm_q_mol)
        self.char_messages = possible_message_states(norm_q_char.shape[0])
        # self.molMessages = molMessages  # Uncomment if you need this field
        
    def __str__(self):
        return (f"ModelDynamics:\n"
                f"q_char: {self.q_char}\n"
                f"j_char: {self.j_char}\n"
                f"q_mol: {self.q_mol}\n"
                f"j_mol: {self.j_mol}\n"
                f"Char Messages: {self.char_messages}\n")
                # Uncomment the following line if mol_messages is used
                # f"Mol Messages: {self.mol_messages}\n")




class AugmentedQTNode(QTNode):
    s_jumps_left: List[int]
    s_jumps_right: List[int]

    def __init__(self, age, left=None, right=None, s_jumps_left=None, s_jumps_right=None):
        super().__init__(age=age, left=left, right=right)
        self.s_jumps_left = s_jumps_left if s_jumps_left is not None else []
        self.s_jumps_right = s_jumps_right if s_jumps_right is not None else []

    def __repr__(self):
        return (f"AugmentedQTNode(age={self.age}, left={self.left!r}, right={self.right!r}, "
                f"s_jumps_left={self.s_jumps_left}, s_jumps_right={self.s_jumps_right})")

    def to_dict(self):
        return {
            "type": "AugmentedQTNode",
            "age": self.age,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
            "s_jumps_left": self.s_jumps_left,
            "s_jumps_right": self.s_jumps_right
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
    
    @classmethod
    def rvs(cls, T: QTNode, nu: float):
        """
        Sample an AugmentedQTNode
        Z ~ q(Z | T, ν)
        """
        # Sample simultaneous jumps to the left and to the right of the root of T        
        s_jumps_left: List[int] = cls.augment_branch(T.age - T.left.age, nu, T.sequence_length) 
        s_jumps_right: List[int] = cls.augment_branch(T.age - T.right.age, nu, T.sequence_length)

        # Recursion
        left = cls.rvs(T.left, nu) if isinstance(T.left, QTNode) else T.left
        right = cls.rvs(T.right, nu) if isinstance(T.right, QTNode) else T.right
        
        new_node = cls(age=T.age, left=left, right=right, s_jumps_left=s_jumps_left, s_jumps_right=s_jumps_right)
        return new_node
    
    def mutate(self, nu:float, ix:int):
        """
        Randomly resamples one lineage
        """
        u = 1. / self.sequence_length
        self.s_jumps_left[ix] = poisson.rvs( (self.age - self.age.left)*u*nu )
        self.s_jumps_right[ix]= poisson.rvs( (self.age - self.age.right)*u*nu )
    
        # Recursion
        # type(self) needed due to class inheritance
        if isinstance(self.left, type(self)):
            self.left.mutate(nu, ix)
        if isinstance(self.right, type(self)):
            self.right.mutate(nu, ix)
    
    def pmf(self, nu: float):
        """
        Returns the pmf q(Z|ν)
        log-likelihood
        """
        u = 1. / self.sequence_length
                
        left_ll_branch = sum(map(lambda s: np.log(poisson.pmf(s, nu*u*(self.age-self.left.age))), self.s_jumps_left))
        right_ll_branch = sum(map(lambda s: np.log(poisson.pmf(s, nu*u*(self.age-self.right.age))), self.s_jumps_right))
        
        # Recursion
        left_ll_tree = self.left.pmf(nu) if isinstance(self.left, QTNode) else 0.0
        right_ll_tree = self.right.pmf(nu) if isinstance(self.right, QTNode) else 0.0
        
        return (right_ll_tree + left_ll_tree + left_ll_branch + right_ll_branch)

    @staticmethod
    def augment_branch(branch_length: float, nu: float, sequence_length: int) -> List[int]:
        u = 1. / sequence_length
        rate = branch_length * nu * u

        def augment(_):
            s_jumps = poisson.rvs(rate)            
            return s_jumps

        results = map(augment, range(sequence_length))
        return list(results)


class PartiallyComputedQTNode(AugmentedQTNode):
    left: Union['QTLeaf', 'ComputedQTNode']
    right: Union['QTLeaf', 'ComputedQTNode']

    def __init__(self, age, left, right, s_jumps_left=None, s_jumps_right=None):
        super().__init__(age=age, left=left, right=right, s_jumps_left=s_jumps_left, s_jumps_right=s_jumps_right)

    def __repr__(self):
        return (f"PartiallyComputedQTNode(age={self.age}, left={self.left!r}, right={self.right!r}, "
                f"s_jumps_left={self.s_jumps_left}, s_jumps_right={self.s_jumps_right})")

    def to_dict(self):
        return {
            "type": "PartiallyComputedQTNode",
            "age": self.age,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
            "s_jumps_left": self.s_jumps_left,
            "s_jumps_right": self.s_jumps_right
        }

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
    
    def to_ComputedQTNode(self, mod: ModelDynamics) -> ComputedQTNode:
        left_time = self.age - self.left.age
        right_time = self.age - self.right.age

        left_char_message = self.left.charMessage if hasattr(self.left, 'charMessage') and self.left.charMessage is not None else self.left.get_char_message(mod.char_messages)
        right_char_message = self.right.charMessage if hasattr(self.right, 'charMessage') and self.right.charMessage is not None else self.right.get_char_message(mod.char_messages)
    
        new_message_list_left = list(map(lambda x, sj: evolve_message(x, left_time, mod, sj), self.left.messageList, self.s_jumps_left))
        new_message_list_right = list(map(lambda x, sj: evolve_message(x, right_time, mod, sj), self.right.messageList, self.s_jumps_right))
    
        new_message_list = list(map(multiply_or_pass, new_message_list_left, new_message_list_right))
    
        new_char_message_left = evolve_char_message(left_char_message, left_time, mod, sum(self.s_jumps_left))
        new_char_message_right = evolve_char_message(right_char_message, right_time, mod, sum(self.s_jumps_right))
        new_character_message = new_char_message_left * new_char_message_right

        return ComputedQTNode(age=self.age,
                              messageList=new_message_list,
                              charMessage=new_character_message,
                              left=self.left,
                              s_jumps_left=self.s_jumps_left,
                              right=self.right,
                              s_jumps_right=self.s_jumps_right)


class ComputedQTNode(AugmentedQTNode):
    """
    ComputedQTNode extends AugmentedQTNode by adding message vectors to each node.
    """
    messageList: List[np.ndarray]             # A list of n-dimensional numpy arrays representing message vectors
    charMessage: np.ndarray                   # A numpy array representing the character message vector

    def __init__(self, age, messageList, charMessage, left=None, right=None, s_jumps_left=None, s_jumps_right=None):
        # Initialize the base class with required properties
        super().__init__(age=age, left=left, right=right, s_jumps_left=s_jumps_left, s_jumps_right=s_jumps_right)
        self.messageList = [np.array(vector) for vector in messageList]
        self.charMessage = np.array(charMessage)

    def __repr__(self):
        return (f"ComputedQTNode(age={self.age}, messageList={self.messageList}, "
                f"charMessage={self.charMessage}, left={self.left!r}, right={self.right!r}, "
                f"s_jumps_left={self.s_jumps_left}, s_jumps_right={self.s_jumps_right})")

    def to_dict(self):
        # Convert numpy arrays to lists for JSON serialization and handle left/right
        node_dict = super().to_dict()
        node_dict.update({
            "messageList": [list(vector) for vector in self.messageList],
            "charMessage": list(self.charMessage)
        })
        return node_dict

    def to_json(self):
        import json
        return json.dumps(self.to_dict(), indent=4)
    
    def update_char_message(self, mod: ModelDynamics):
        """
        TODO implement this without a side-effect
        Recomputing the character message in place as Python's object are passed
        by reference
        """
        if (isinstance(self.left, QTLeaf) and isinstance(self.right, QTLeaf)):
            self._update_char_message(mod)   
        elif (isinstance(self.left, QTLeaf) and isinstance(self.right, ComputedQTNode)):
            self.right.update_char_message(mod)
            self._update_char_message(mod)   
        elif (isinstance(self.left, ComputedQTNode) and isinstance(self.right, QTLeaf)):
            self.left.update_char_message(mod)
            self._update_char_message(mod)   
        elif (isinstance(self.left, ComputedQTNode) and isinstance(self.right, ComputedQTNode)):
            self.left.update_char_message(mod)
            self.right.update_char_message(mod)
            self._update_char_message(mod)    
        else:   
            raise Exception("Unexpected tree encountered: " + self)
        
    def _update_char_message(self, mod: ModelDynamics):
        left_time:float = self.age - self.left.age
        right_time:float = self.age - self.right.age
        
        # If 'left' is a QTLeaf, we need to compute the initial character message based on the number of possible characters; otherwise, we simply retrieve the value.
        left_char_message = self.left.charMessage if hasattr(self.left, 'charMessage') and self.left.charMessage is not None else self.left.get_char_message(mod.char_messages)
        right_char_message = self.right.charMessage if hasattr(self.right, 'charMessage') and self.right.charMessage is not None else self.right.get_char_message(mod.char_messages)
        
        new_char_message_left=evolve_char_message(left_char_message, left_time, mod, sum(self.s_jumps_left))
        new_char_message_right=evolve_char_message(right_char_message, right_time, mod, sum(self.s_jumps_right))
        new_character_message=new_char_message_left * new_char_message_right
        
        self.charMessage=new_character_message
        
    def update_message(self, mod: ModelDynamics, ix: int):
        """
        TODO implement this without a side-effect
        Recomputing the character message in place as Python's object are passed
        by reference
        """
        if (isinstance(self.left, QTLeaf) and isinstance(self.right, QTLeaf)):
            self._update_message(mod, ix)   
        elif (isinstance(self.left, QTLeaf) and isinstance(self.right, ComputedQTNode)):
            self.right.update_message(mod, ix)
            self._update_message(mod, ix)   
        elif (isinstance(self.left, ComputedQTNode) and isinstance(self.right, QTLeaf)):
            self.left.update_message(mod, ix)
            self._update_message(mod, ix)   
        elif (isinstance(self.left, ComputedQTNode) and isinstance(self.right, ComputedQTNode)):
            self.left.update_message(mod, ix)
            self.right.update_message(mod, ix)
            self._update_message(mod, ix)    
        else:   
            raise Exception("Unexpected tree encountered: " + self)
        
    def _update_message(self, mod: ModelDynamics,ix: int):
        left_time:float = self.age - self.left.age
        right_time:float = self.age - self.right.age
        
        new_message_left = evolve_message(self.left.messageList[ix], left_time, mod, self.s_jumps_left[ix])
        new_message_right = evolve_message(self.right.messageList[ix], right_time, mod, self.s_jumps_right[ix])
        self.messageList[ix] = multiply_or_pass(new_message_left, new_message_right)

        
    def update_message_list(self, mod: ModelDynamics):
        """
        TODO implement this without a side-effect
        Recomputing the character message in place as Python's object are passed
        by reference
        """
        if (isinstance(self.left, QTLeaf) and isinstance(self.right, QTLeaf)):
            self._update_message_list(mod)   
        elif (isinstance(self.left, QTLeaf) and isinstance(self.right, ComputedQTNode)):
            self.right.update_message_list(mod)
            self._update_message_list(mod)   
        elif (isinstance(self.left, ComputedQTNode) and isinstance(self.right, QTLeaf)):
            self.left.update_message_list(mod)
            self._update_message_list(mod)   
        elif (isinstance(self.left, ComputedQTNode) and isinstance(self.right, ComputedQTNode)):
            self.left.update_message_list(mod)
            self.right.update_message_list(mod)
            self._update_message_list(mod)    
        else:   
            raise Exception("Unexpected tree encountered: " + self)
        
    def _update_message_list(self: ComputedQTNode, mod: ModelDynamics):
        left_time:float = self.age - self.left.age
        right_time:float = self.age - self.right.age
        
        new_message_list_left = list(map(lambda x, sj: evolve_message(x, left_time, mod, sj), self.left.messageList, self.s_jumps_left))
        new_message_list_right = list(map(lambda x, sj: evolve_message(x, right_time, mod, sj), self.right.messageList, self.s_jumps_right))
        new_message_list = list(map(multiply_or_pass, new_message_list_left, new_message_list_right))

        self.messageList=new_message_list


    def compute_tree_loglik(self) -> float:
        log_weight_list: List[float] = list(map(compute_message_log_likelihood, self.messageList))
        log_weight_char: float = compute_message_log_likelihood(self.charMessage)
        
        # Add log_weight_char to each element in log_weight_list
        l = [weight + log_weight_char for weight in log_weight_list][0]
        return l
    
    @classmethod
    def from_AugmentedQTNode(cls, tree: AugmentedQTNode, mod):
        """
        The ComputedQTNode here is the initial tree that has been collapsed into a single leaf
        that has the correct messages needed for the likelihood computation
        """
        if isinstance(tree.left, QTLeaf) and isinstance(tree.right, QTLeaf):
            new_tree = PartiallyComputedQTNode(
                age=tree.age, 
                left=tree.left, 
                right=tree.right, 
                s_jumps_left=tree.s_jumps_left, 
                s_jumps_right=tree.s_jumps_right
            )
            return new_tree.to_ComputedQTNode(mod)
        elif isinstance(tree.left, QTLeaf) and isinstance(tree.right, AugmentedQTNode):
            right_tree = cls.from_AugmentedQTNode(tree.right, mod)
            new_tree = PartiallyComputedQTNode(
                age=tree.age, 
                left=tree.left, 
                right=right_tree, 
                s_jumps_left=tree.s_jumps_left, 
                s_jumps_right=tree.s_jumps_right
            )
            return new_tree.to_ComputedQTNode(mod)
        elif isinstance(tree.left, AugmentedQTNode) and isinstance(tree.right, QTLeaf):
            left_tree = cls.from_AugmentedQTNode(tree.left, mod)
            new_tree = PartiallyComputedQTNode(
                age=tree.age, 
                left=left_tree, 
                right=tree.right, 
                s_jumps_left=tree.s_jumps_left, 
                s_jumps_right=tree.s_jumps_right
            )
            return new_tree.to_ComputedQTNode(mod)
        elif isinstance(tree.left, AugmentedQTNode) and isinstance(tree.right, AugmentedQTNode):
            left_tree = cls.from_AugmentedQTNode(tree.left, mod)
            right_tree = cls.from_AugmentedQTNode(tree.right, mod)
            new_tree = PartiallyComputedQTNode(
                age=tree.age, 
                left=left_tree, 
                right=right_tree, 
                s_jumps_left=tree.s_jumps_left, 
                s_jumps_right=tree.s_jumps_right
            )
            return new_tree.to_ComputedQTNode(mod)
        else:
            raise Exception("Unexpected tree structure encountered")    


