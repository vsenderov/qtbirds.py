from .QTTree import QTNode, QTLeaf
from .AugmentedQTTree import AugmentedQTNode, PartiallyComputedQTNode, ComputedQTNode, ModelDynamics
from typing import Optional, Dict, Tuple, List
import numpy as np
from scipy.stats import poisson, gamma, norm
import random
import copy

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
    
    # TODO probably we don't want to store nu or the model dynamics here
    mod = ModelDynamics(norm_q_char = norm_q_char,
                        lam = lam,
                        norm_q_mol = norm_q_mol,
                        mu = mu)
    
    print(mod)
    
    #Z, _ = data_augment_tree(tree, mod)
    #M: ComputedQTNode = compute_messages_rec(Z, mod)
    # Sample the augmented tree
    #l:float = compute_tree_loglik(M)
    Z = AugmentedQTNode.rvs(T = tree, nu = nu)
    M = ComputedQTNode.from_AugmentedQTNode(Z, mod=mod)
    l:float = M.compute_tree_loglik()

    ##
    # Proposal loop
    ##
    
    lambda_samples = [lam]
    mu_samples = [mu]
    nu_samples = [nu]
    p_samples = []
    
    accept_count = 0
    reject_count = 0
    samples_count = 0
    while samples_count < samples:
        case: int = random.randint(1, 3) 
        match case:
            case 1:
                """
                Proposing a new value for lambda
                - we only recompute the character message
                """
                #lam_new, q_ratio = propose_move(lam, prior)
                # Propose a move form the prior
                lam_new = gamma.rvs(a=prior['lam']['shape'], scale=prior['lam']['scale'])
                
                # TODO propose from a normal move, but pay attention to what happens around 0,
                # since negative numbers can't be proposed, therefore we need to propose 0
                # but we need to put all the mass for <= 0
                
                mod_new = ModelDynamics(norm_q_char = norm_q_char,
                        lam = lam_new,
                        norm_q_mol = norm_q_mol,
                        mu = mu)
                
                M_new = copy.deepcopy(M)
                M_new.update_char_message(mod_new)
                l_new = M_new.compute_tree_loglik()
                
                a = random.uniform(0, 1)
                
                # Here is the full expression:
                #
                # L(λ', μ, ν, Ζ) p(Z|ν) p(λ') q(λ|λ')
                # -------------  ------ ---- ----
                # L(λ, μ, ν, Ζ)  p(Z|ν) p(λ)  q(λ'|λ)
                # 
                # Since we moved only λ, P(Z|ν) cancels, and since we
                # resampled from the prior q(λ|λ') = p(λ)
                # so the move cancels with the prior
                A = min(1., np.exp(l_new)/np.exp(l))
                if (A > a): # accept    
                    accept_count+=1
                    lambda_samples.append(lam_new)
                    mu_samples.append(mu)
                    nu_samples.append(nu)
                    M=M_new # TODO do we need to garbage collect?
                    lam=lam_new 
                    mod=mod_new # TODO do we need to garbage collect?
                    l=l_new
                    samples_count+=1
                else:
                    reject_count+=1
            case 2:
                """
                Proposing a new value for mu
                - we update the molecular messages but not the character message
                """
                mu_new = gamma.rvs(a=prior['mu']['shape'], scale=prior['mu']['scale'])
                
                # TODO propose from a normal move, but pay attention to what happens around 0,
                # since negative numbers can't be proposed, therefore we need to propose 0
                # but we need to put all the mass for <= 0
                
                mod_new = ModelDynamics(norm_q_char = norm_q_char,
                        lam = lam,
                        norm_q_mol = norm_q_mol,
                        mu = mu_new)
                
                M_new = copy.deepcopy(M)
                M_new.update_message_list(mod_new)
                l_new = M_new.compute_tree_loglik()
                
                a = random.uniform(0, 1)
                
                # Here is the full expression:
                #
                # L(λ, μ', ν, Ζ) p(Z|ν) p(μ') q(μ|μ')
                # -------------  ------ ---- ----
                # L(λ, μ, ν, Ζ)  p(Z|ν) p(μ)  q(μ'|μ)
                # 
                # Since we moved only μ, P(Z|ν) cancels, and since we
                # resampled from the prior q(μ|μ') = p(μ)
                # so the move cancels with the prior
                A = min(1., np.exp(l_new)/np.exp(l))
                if (A > a): # accept
                    accept_count+=1
                    lambda_samples.append(lam)
                    mu_samples.append(mu_new)
                    nu_samples.append(nu)
                    M=M_new # TODO do we need to garbage collect?
                    mu=mu_new 
                    mod=mod_new # TODO do we need to garbage collect?
                    l=l_new
                    samples_count+=1
                else:
                    reject_count+=1
            case 3:
                """
                Proposing a new value for nu
                - we update Z
                - then we recompute all messages
                """
                nu_new = gamma.rvs(a=prior['mu']['shape'], scale=prior['mu']['scale'])
                
                # TODO propose from a normal move, but pay attention to what happens around 0,
                # since negative numbers can't be proposed, therefore we need to propose 0
                # but we need to put all the mass for <= 0
                
                Z_new = AugmentedQTNode.rvs(T = tree, nu = nu_new)
                M_new = ComputedQTNode.from_AugmentedQTNode(Z_new, mod=mod)
                l_new = M_new.compute_tree_loglik()
                
                a = random.uniform(0, 1)
                
                # Here is the full expression:
                #
                # L(λ, μ, ν', Ζ') p(Z'|ν') p(v') q(μ|μ')
                # -------------  ------ ---- ----
                # L(λ, μ, ν, Ζ)  p(Z|ν) p(ν)  q(μ'|μ)
                # 
                # since resampled from the prior q(ν|ν') = p(ν)
                # so the move cancels with the prior
                # but we need to compute the weight ratio for the latent vars
                
                A = min(1., np.exp(l_new)/np.exp(l) * np.exp(Z_new.pmf(nu_new))/np.exp(Z.pmf(nu)))
                if (A > a): # accept    
                    accept_count+=1
                    lambda_samples.append(lam)
                    mu_samples.append(mu)
                    nu_samples.append(nu_new)
                    M=M_new # TODO do we need to garbage collect?
                    nu=nu_new
                    Z=Z_new # TODO do we need to garbage collect?
                    samples_count+=1
                else:
                    reject_count+=1
            case 4:
                """
                Moving the hidden state without touching the vars
                This is TURNED OFF right now as we change Z in case 3
                """
                M_new = copy.deepcopy(M)
                ix:int = random.randint(0, Z_new.sequence_length - 1)
                print("case ", case, ix)
                M_new.mutate(nu, ix) # will randomly resample one lineage
                M_new.update_message(mod, ix)
                M_new.update_char_message(mod) # needs to be updated also
                l_new = M_new.compute_tree_loglik()
                

    print("A/R ratio:", accept_count/reject_count)    
    return lambda_samples, mu_samples, nu_samples, p_samples      






