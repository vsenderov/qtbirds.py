from .QTTree import QTNode, AugmentedQTNode, PartiallyComputedQTNode, QTLeaf, ComputedQTNode, find_sequence_length
from typing import Optional, Dict, Tuple, List
import numpy as np
from .linear_algebra import calc_jump_matrix, possible_message_states
from scipy.stats import poisson, gamma
from scipy.linalg import expm
import random



class ModelDynamics:
    q_char: np.ndarray
    j_char: np.ndarray
    q_mol: np.ndarray
    j_mol: np.ndarray
    nu: float
    sequence_length: int
    char_messages: List[List[float]]  # Possible initial character messages (depends on the number of character states)
    # molMessages: List[List[float]]  # Uncomment if you need this field

    def __init__(self, lam: float, norm_q_char: np.ndarray, mu: float, norm_q_mol: np.ndarray, nu: float, sequence_length: int):
        self.q_char = lam*norm_q_char
        self.j_char = calc_jump_matrix(norm_q_char) 
        self.q_mol = mu*norm_q_mol
        self.j_mol = calc_jump_matrix(norm_q_mol)
        self.nu = nu
        self.char_messages = possible_message_states(norm_q_char.shape[0])
        self.sequence_length = sequence_length
        # self.molMessages = molMessages  # Uncomment if you need this field
        
    def __str__(self):
        return (f"ModelDynamics:\n"
                f"q_char: {self.q_char}\n"
                f"j_char: {self.j_char}\n"
                f"q_mol: {self.q_mol}\n"
                f"j_mol: {self.j_mol}\n"
                f"Nu: {self.nu}\n"
                f"Char Messages: {self.char_messages}\n"
                f"Sequence length: {self.sequence_length}\n")
                # Uncomment the following line if mol_messages is used
                # f"Mol Messages: {self.mol_messages}\n")

def qt_mcmc(
            tree: QTNode,
            label: str = "No label",
            prior: Optional[Dict[str, Dict[str, float]]] = None,
            norm_q_mol: Optional[np.ndarray] = None,
            norm_q_char: Optional[np.ndarray] = None,
            samples: int = 100,
            burnin: int = 5000,
            chains: int = 6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    MCMC Inference for the QT Model

    :param tree: A phylogenetic tree that has QT-type data in its leafs.  The type
        is implemented as a class in QTTree.py.  I have put type-annotations to make
        it clear what it is.
                
    :param label:
    
    :param prior: The hyper-parameters of the Gamma distribution for λ, μ, ν (Gamma) 
        and p (Beta) e.g. NOTE(vsenderov, 24-04-22: p is ignored for now)                
        {
            'lam': {'shape': 1.0, 'scale': 0.5},
            'mu': {'shape': 1.0, 'scale': 0.5},
            'nu': {'shape': 1.0, 'scale': 0.5},
            'p': {'pa': 0.5, 'pb': 0.5}
        }

    :param norm_q_mol: The normalized Q-matrix for the molecular process 
        (default Jukes-Cantor matrix provided)

    :param norm_q_char: The normalized Q-matrix for the phenotypic process
        (default binary Mk matrix provided)

    :param samples: How many samples to print, excluding burnin.
    :param burnin: How many samples at the beginning of the chain to drop.
    :pram chains: How many chains to run? NOTE(vsenderov, 24-04-22: only 1 supported for now)
    """
    
    ###
    # Housekeeping
    ###
    
    if prior is None:
        print("Using default prior:")
        prior = {
            'lam': {'shape': 1.0, 'scale': 0.5},
            'mu': {'shape': 1.0, 'scale': 0.5},
            'nu': {'shape': 1.0, 'scale': 0.5},
            'p': {'pa': 0.5, 'pb': 0.5}
        }
    else:
        print("Using provided prior:")
        
    print(prior)
        
    if norm_q_mol is None:
        print("Using default molecular model (JC):")
        norm_q_mol = np.array( [[-1., 1/3, 1/3, 1/3],
                                [1/3, -1., 1/3, 1/3],
                                [1/3, 1/3, -1., 1/3],
                                [1/3, 1/3, 1/3, -1.]])
    else:
        print("Using provided molecular model:")
        
    print(norm_q_mol)
        
    if norm_q_char is None:
        print("Using default phenotype model (binary Mk):")
        norm_q_char = np.array([[-1.,  1.],
                                [ 1., -1.]])
    else:
        print("Using provided phenotype model:")
        
    print(norm_q_char)
    
    ###
    # MCMC Initialization
    ###
    
    lam = gamma.rvs(a=prior['lam']['shape'], scale=prior['lam']['scale'])
    mu = gamma.rvs(a=prior['lam']['shape'], scale=prior['lam']['scale'])
    nu = gamma.rvs(a=prior['lam']['shape'], scale=prior['lam']['scale'])
    
    mod = ModelDynamics(norm_q_char = norm_q_char,
                        lam = lam,
                        norm_q_mol = norm_q_mol,
                        mu = mu,
                        nu = nu,
                        sequence_length=find_sequence_length(tree))
    
    print(mod)
    
    Z, _ = data_augment_tree(tree, mod)
    
    M: ComputedQTNode = compute_messages_rec(Z, mod)
    l:float = compute_tree_loglik(M)

    ##
    # Proposal loop
    ##
    
    lambda_samples = [lam]
    mu_samples = [mu]
    nu_samples = [nu]
    p_samples = []
    
    samples_count = 0
    while samples_count < 10:
        case: int = random.randint(1, 3)
        case = 1   
        match case:
            case 1:
                """
                Proposing a new value for lambda
                - we only recompute the character message
                """
                lam_new, q_ratio = propose_prior(lam, prior)
                # TODO propose from a normal move, but pay attention to what happens around 0,
                # since negative numbers can't be proposed, therefore we need to propose 0
                # but we need to put all the mass for <= 0
                mod_new = ModelDynamics(norm_q_char = norm_q_char,
                        lam = lam_new,
                        norm_q_mol = norm_q_mol,
                        mu = mu,
                        nu = nu,
                        sequence_length=find_sequence_length(tree))
                
                M_new = M
                update_char_message_rec(M_new, mod_new) # side effect on M_new, not sure if this is the right way to go
                l_new = compute_tree_loglik(M_new)
                
                a = random.uniform(0, 1)
                A = min(1., np.exp(l_new)/np.exp(l)*q_ratio)
                if (A > a): # accept    
                    lambda_samples.append(lam_new)
                    mu_samples.append(mu)
                    nu_samples.append(nu)
                    M=M_new # TODO turn this into a lambda to avoid overwriting
                    lam=lam_new # TODO not functional
                    mod=mod_new # TODO not functional
                    samples_count+=1
                    
    
    return lambda_samples, mu_samples, nu_samples, p_samples      


def propose_prior(prev: float, prior):
    """"
    Proposes from the prior
    """
    newval = gamma.rvs(a=prior['lam']['shape'], scale=prior['lam']['scale'])
    prev_w = gamma.pdf(prev, a=prior['lam']['shape'], scale=prior['lam']['scale'])
    newval_w = gamma.pdf(newval, a=prior['lam']['shape'], scale=prior['lam']['scale'])

    q_ratio = prev_w / newval_w

    return newval, q_ratio
    

def compute_tree_loglik(M: ComputedQTNode) -> float:
    log_weight_list: List[float] = list(map(compute_message_log_likelihood, M.messageList))
    log_weight_char: float = compute_message_log_likelihood(M.charMessage)
    
    # Add log_weight_char to each element in log_weight_list
    l = [weight + log_weight_char for weight in log_weight_list][0]
    return l
    

def data_augment_tree(tree: QTNode, mod: ModelDynamics) -> Tuple[AugmentedQTNode, float]:
    """
    Samples the latent s-jumps along tree
    
        Z ~ q(Z | ν) 
        
    !!! NB: they only depend on ν !!!
    
    :param tree: Unaugmented tree
    :param mod: The model dynamics, however we only depend on mu and nu for this procedure!
    
    Return values
        :Z: a data augmented tree that has the s-jumps on each branch
        :logw: the log-likelihood of such an augmentation
    """
    # Assuming data_augment_branch correctly returns a tuple (s_jumps, logw)
    s_jumps_left, logw_left = data_augment_branch(tree.left.age, mod.sequence_length, mod.nu)
    s_jumps_right, logw_right = data_augment_branch(tree.right.age, mod.sequence_length, mod.nu)

    # Recursive augmentation if the node is not a leaf
    # however beware of Python's weird evaluation rules
    left_augmented, logw_left_tree = (data_augment_tree(tree.left, mod) if isinstance(tree.left, QTNode) else (tree.left, 0))
    right_augmented, logw_right_tree = (data_augment_tree(tree.right, mod) if isinstance(tree.right, QTNode) else (tree.right, 0))

    new_tree = AugmentedQTNode(age=tree.age,
                              left=left_augmented,
                              s_jumps_left=s_jumps_left, 
                              right=right_augmented,
                              s_jumps_right=s_jumps_right)
    
    return new_tree, logw_left + logw_right + logw_right_tree + logw_left_tree
    
    
def data_augment_branch(branch_length: float, sequence_length: int, nu: float) -> Tuple[List[int], float]:
    u = 1./float(sequence_length)
    rate = branch_length * nu * u
    
    # Function to be applied to each element in the sequence
    def augment(_):
        s_jumps = poisson.rvs(rate)
        prob = poisson.pmf(s_jumps, rate)
        loglik = np.log(prob)  # Calculating log likelihood
        return s_jumps, loglik

    results = map(augment, range(sequence_length))
    s_jumps_list, loglik_list = map(list, zip(*results))  # Unzipping the list of tuples
    total_loglik = sum(loglik_list)  # Summing up the log likelihoods
    return s_jumps_list, total_loglik


def compute_messages_rec(
    tree: AugmentedQTNode,
    mod: ModelDynamics
    ) -> ComputedQTNode:
    """
    The ComputedQTNode here is the initial tree that has been collapsed into a single leaf
    that has the correct messages needed for the likelihood computation
    
    """
    if (isinstance(tree.left, QTLeaf) and isinstance(tree.right, QTLeaf)):
        new_tree = PartiallyComputedQTNode(
            left=tree.left,
            s_jumps_left=tree.s_jumps_left,
            right=tree.right,
            s_jumps_right=tree.s_jumps_right,
            age=tree.age
        )
        return coalesce(new_tree, mod)
    elif (isinstance(tree.left, QTLeaf) and isinstance(tree.right, AugmentedQTNode)):
        right_tree = compute_messages_rec(tree.right, mod)
        new_tree = PartiallyComputedQTNode(
            left=tree.left,
            s_jumps_left=tree.s_jumps_left,
            right=right_tree,
            s_jumps_right=tree.s_jumps_right,
            age=tree.age
        )
        return coalesce(new_tree, mod)
    elif (isinstance(tree.left, AugmentedQTNode) and isinstance(tree.right, QTLeaf)):
        left_tree = compute_messages_rec(tree.left, mod)
        new_tree = PartiallyComputedQTNode(
            left=left_tree,
            s_jumps_left=tree.s_jumps_left,
            right=tree.right,
            s_jumps_right=tree.s_jumps_right,
            age=tree.age
        )
        return coalesce(new_tree, mod)
    elif (isinstance(tree.left, AugmentedQTNode) and isinstance(tree.right, AugmentedQTNode)):
        left_tree = compute_messages_rec(tree.left, mod)
        right_tree = compute_messages_rec(tree.right, mod)
        new_tree = PartiallyComputedQTNode(
            left=left_tree,
            s_jumps_left=tree.s_jumps_left,
            right=right_tree,
            s_jumps_right=tree.s_jumps_right,
            age=tree.age
        )
        return coalesce(new_tree, mod)
    else:   
        raise Exception("Unexpected tree encountered: " + tree)
        
        

def update_char_message_rec(
    tree: ComputedQTNode,
    mod: ModelDynamics
    ):
    """
    TODO implement this without a side-effect
    Recomputing the character message in place as Python's object are passed
    by reference
    """
    if (isinstance(tree.left, QTLeaf) and isinstance(tree.right, QTLeaf)):
        update_char_message(tree, mod)   
    elif (isinstance(tree.left, QTLeaf) and isinstance(tree.right, ComputedQTNode)):
        update_char_message_rec(tree.right, mod)
        update_char_message(tree, mod)
    elif (isinstance(tree.left, ComputedQTNode) and isinstance(tree.right, QTLeaf)):
        update_char_message_rec(tree.left, mod)
        update_char_message(tree, mod)
    elif (isinstance(tree.left, ComputedQTNode) and isinstance(tree.right, ComputedQTNode)):
        update_char_message_rec(tree.left, mod)
        update_char_message_rec(tree.right, mod)
        update_char_message(tree, mod) 
    else:   
        raise Exception("Unexpected tree encountered: " + tree)


def update_char_message(tree: ComputedQTNode, mod: ModelDynamics):
    left_time:float = tree.age - tree.left.age
    right_time:float = tree.age - tree.right.age
    
    # If 'left' is a QTLeaf, we need to compute the initial character message based on the number of possible characters; otherwise, we simply retrieve the value.
    left_char_message = tree.left.charMessage if hasattr(tree.left, 'charMessage') and tree.left.charMessage is not None else tree.left.get_char_message(mod.char_messages)
    right_char_message = tree.right.charMessage if hasattr(tree.right, 'charMessage') and tree.right.charMessage is not None else tree.right.get_char_message(mod.char_messages)
    
    new_char_message_left=evolve_char_message(left_char_message, left_time, mod, sum(tree.s_jumps_left))
    new_char_message_right=evolve_char_message(right_char_message, left_time, mod, sum(tree.s_jumps_right))
    new_character_message=new_char_message_left * new_char_message_right
    
    tree.charMessage=new_character_message


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


def coalesce(tree: PartiallyComputedQTNode, mod: ModelDynamics) -> ComputedQTNode:
    left_time:float = tree.age - tree.left.age
    right_time:float = tree.age - tree.right.age

    # If 'left' is a QTLeaf, we need to compute the initial character message based on the number of possible characters; otherwise, we simply retrieve the value.
    left_char_message = tree.left.charMessage if hasattr(tree.left, 'charMessage') and tree.left.charMessage is not None else tree.left.get_char_message(mod.char_messages)
    right_char_message = tree.right.charMessage if hasattr(tree.right, 'charMessage') and tree.right.charMessage is not None else tree.right.get_char_message(mod.char_messages)
    
    new_message_list_left = list(map(lambda x, sj: evolve_message(x, left_time, mod, sj), tree.left.messageList, tree.s_jumps_left))
    new_message_list_right = list(map(lambda x, sj: evolve_message(x, right_time, mod, sj), tree.right.messageList, tree.s_jumps_right))
    
    new_message_list = list(map(multiply_or_pass, new_message_list_left, new_message_list_right))
    
    new_char_message_left=evolve_char_message(left_char_message, left_time, mod, sum(tree.s_jumps_left))
    new_char_message_right=evolve_char_message(right_char_message, left_time, mod, sum(tree.s_jumps_right))
    new_character_message=new_char_message_left * new_char_message_right

    return ComputedQTNode(age=tree.age,
                          messageList=new_message_list,
                          charMessage=new_character_message,
                          left = tree.left,
                          s_jumps_left = tree.s_jumps_left,
                          right = tree.right,
                          s_jumps_right=tree.s_jumps_right)
    

    
def compute_message_log_likelihood(mes: ComputedQTNode) -> float:
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