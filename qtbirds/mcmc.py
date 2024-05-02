# Suggestion: no spaces around naming
# but spaces around assignment
# pip install black // use dry mode
# -l to specify line length (120 for example)
# black --diff --line-length 120 qtbirds/mcmc.py
# black --line-length 120 qtbirds/mcmc.py

from .QTTree import QTNode, QTLeaf
from .AugmentedQTTree import AugmentedQTNode, PartiallyComputedQTNode, ComputedQTNode, ModelDynamics
from typing import Optional, Dict, Tuple, List
import numpy as np
from scipy.stats import gamma, norm
import random
import copy
import csv

def qt_mcmc(
            tree: QTNode,
            label: str = "no-label",
            prior: Optional[Dict[str, Dict[str, float]]] = None,
            norm_q_mol: Optional[np.ndarray] = None,
            norm_q_char: Optional[np.ndarray] = None,
            samples: int = 100,
            burnin: int = 5000,
            thinning: int = 1000,
            random: bool = True,
            chains: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
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
    lam_logw = np.log(gamma.pdf(lam, a=prior['lam']['shape'], scale=prior['lam']['scale']))

    mu = gamma.rvs(a=prior['mu']['shape'], scale=prior['mu']['scale'])
    mu_logw = np.log(gamma.pdf(mu, a=prior['mu']['shape'], scale=prior['mu']['scale']))
    
    nu = gamma.rvs(a=prior['nu']['shape'], scale=prior['nu']['scale'])
    nu_logw = np.log(gamma.pdf(nu, a=prior['nu']['shape'], scale=prior['nu']['scale']))

    mod = ModelDynamics(norm_q_char = norm_q_char,
                        lam = lam,
                        norm_q_mol = norm_q_mol,
                        mu = mu)
    
    print(mod)
    
    Z = AugmentedQTNode.rvs(T = tree, nu = nu)
    M = ComputedQTNode.from_AugmentedQTNode(Z, mod=mod)
    l:float = M.compute_tree_loglik()

    ##
    # Proposal loop
    ##
    
    lambda_samples = []
    accept_count_lam = 0
    reject_count_lam = 0
    mu_samples = []
    accept_count_mu = 0
    reject_count_mu = 0
    nu_samples = []
    accept_count_nu = 0
    reject_count_nu = 0
    p_samples = []
    case = 0
    filename = "mcmc-output_" + label + ".csv"
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
    
        # Write the header
        header = ['lambda', 'mu', 'nu', 'p']
        writer.writerow(header)
        print("Samples are continuously saved to " + filename)
        print("Using random scan: " + random)
        
        # Now the file is ready for further data writing.
        # For example:
        # writer.writerow([0.1, 0.2, 0.3, 0.4])  # This would write one row of sample data
    
        total = samples*thinning + burnin
        for i in range(total):
            if random:
                case = random.randint(0, 2) # Random scan Metropolis-Hastings
            else:
                case = (case + 1) % 3 # Component-wise updating 
            match case:
                case 0:
                    """
                    Proposing a new value for lambda
                    - we only recompute the character message
                    """
                    # Uncomment to propose a move form the prior 
                    # lam_new = gamma.rvs(a=prior['lam']['shape'], scale=prior['lam']['scale'])
                    
                    # Gaussian shift
                    lam_new = norm.rvs(loc=lam, scale=prior['lam']['scale'])
                    
                    if (lam_new>=0.):
                        lam_new_logw=np.log(gamma.pdf(lam_new, a=prior['lam']['shape'], scale=prior['lam']['scale']))
                        mod_new = ModelDynamics(norm_q_char=norm_q_char,
                                lam=lam_new,
                                norm_q_mol=norm_q_mol,
                                mu=mu)
                        M_new = copy.deepcopy(M)
                        M_new.update_char_message(mod_new)
                        l_new = M_new.compute_tree_loglik()
                        
                        # Here is the full expression:
                        #
                        # L(T| λ', μ, ν, Ζ) p(Z|ν) p(λ') q(λ|λ')
                        #   -------------  ------   ---- ----
                        # L(T| λ, μ, ν, Ζ)  p(Z|ν) p(λ)  q(λ'|λ)
                        # 
                        # P(Z|v) cancels because no change 
                        # q(λ|λ') also cancels because symmetric move
                        # need to take care of likelihood and prior weight
                        
                        a = random.uniform(0, 1)
                        A = l_new - l + lam_new_logw - lam_logw
                        if A > np.log(a): # accept    
                            if i >= burnin:
                                accept_count_lam += 1
                                
                            lam=lam_new 
                            lam_logw = lam_new_logw
                            M=M_new # TODO do we need to garbage collect?
                            mod=mod_new # TODO do we need to garbage collect?
                            l=l_new
                        else:
                            if i >= burnin:
                                reject_count_lam += 1
                case 1:
                    """
                    Proposing a new value for mu
                    - we update the molecular messages but not the character message
                    """
                    # Redrawing from prior
                    # mu_new = gamma.rvs(a=prior['mu']['shape'], scale=prior['mu']['scale'])
                    
                    # Gaussian shift
                    mu_new = norm.rvs(loc=mu, scale=prior['mu']['scale'])

                    if (mu_new >= 0.):
                        mu_new_logw=np.log(gamma.pdf(mu_new, a=prior['mu']['shape'], scale=prior['mu']['scale']))
                    
                        mod_new = ModelDynamics(norm_q_char=norm_q_char,
                                lam=lam,
                                norm_q_mol=norm_q_mol,
                                mu=mu_new)
                        
                        M_new = copy.deepcopy(M)
                        M_new.update_message_list(mod_new)
                        l_new = M_new.compute_tree_loglik()
                                            
                        # Here is the full expression:
                        #
                        # L(λ, μ', ν, Ζ) p(Z|ν) p(μ') q(μ|μ')
                        # -------------  ------ ---- ----
                        # L(λ, μ, ν, Ζ)  p(Z|ν) p(μ)  q(μ'|μ)
                        # 
                        # P(Z|v) cancels because no change 
                        # q(μ|μ') also cancels because symmetric move
                        # need to take care of likelihood and prior weight
                        
                        a = random.uniform(0, 1)
                        A = l_new - l + mu_new_logw - mu_logw
                        if A > np.log(a): # accept
                            if i >= burnin:
                                accept_count_mu += 1
                            
                            M = M_new # TODO do we need to garbage collect?
                            mu = mu_new
                            mu_logw = mu_new_logw
                            mod = mod_new # TODO do we need to garbage collect?
                            l = l_new
                        else:
                            if i>= burnin:
                                reject_count_mu += 1
                case 2:
                    """
                    Proposing a new value for nu
                    - we update Z
                    - then we recompute all messages
                    """
                    # Sample from prior
                    # nu_new = gamma.rvs(a=prior['mu']['shape'], scale=prior['mu']['scale'])
                    
                    # Gaussian move
                    nu_new = norm.rvs(loc=nu, scale=prior['nu']['scale'])

                    if (nu_new >= 0.):
                        nu_new_logw=np.log(gamma.pdf(nu_new, a=prior['nu']['shape'], scale=prior['nu']['scale']))

                        Z_new = AugmentedQTNode.rvs(T = tree, nu = nu_new)
                        M_new = ComputedQTNode.from_AugmentedQTNode(Z_new, mod=mod)
                        l_new = M_new.compute_tree_loglik()
                        
                        # Here is the full expression:
                        #
                        # p(T| λ, μ, Ζ')    p(Z'|ν') p(v') q(ν| ν')
                        #  -------------     ------   ----    ----
                        # p(T| λ, μ, Ζ)    p(Z|ν)  p(ν)  q(ν'|ν )
                        # 
                        # q - cancels because of symmetric move
                        # p(v) - doesn't cancel
                        # p(Z|v) - doesn't cancel
                        
                        a = random.uniform(0, 1)
                        A = l_new - l + nu_new_logw - nu_logw + Z_new.pmf(nu_new) - Z.pmf(nu)
                        
                        if A > np.log(a): # accept    
                            if i >= burnin:
                                accept_count_nu += 1
                                
                            M = M_new # TODO do we need to garbage collect?
                            Z = Z_new # TODO do we need to garbage collect?
                            l = l_new
                            nu = nu_new
                            nu_logw = nu_new_logw
                        else:
                            if i >= burnin:
                                reject_count_nu += 1
                            
                case 4:
                    """
                    Moving the hidden state without touching the vars
                    This is TURNED OFF right now as we change Z in case 3
                    
                    perhaps we can do case 3 backwards:
                        - we propose a new value  Z -> Z'
                        - then, we try to accept a new v'|Z'
                        - this way we changed Z only slightly
                    """
                    raise Exception("This case should never occur")
                    M_new = copy.deepcopy(M)
                    ix:int = random.randint(0, Z_new.sequence_length - 1)
                    print("case ", case, ix)
                    M_new.mutate(nu, ix) # will randomly resample one lineage
                    M_new.update_message(mod, ix)
                    M_new.update_char_message(mod) # needs to be updated also
                    l_new = M_new.compute_tree_loglik()
            
            if (i >= burnin) and (i % thinning == 0):
                writer.writerow([lam, mu, nu, None])  # This would write one row of sample data
                lambda_samples.append(lam)
                mu_samples.append(mu)
                nu_samples.append(nu)
                p_samples.append(None)

    print("A/R ratio λ (excluding burnin):", accept_count_lam/reject_count_lam)
    print("A/R ratio μ (excluding burnin):", accept_count_mu/reject_count_mu)
    print("A/R ratio ν (excluding burnin):", accept_count_nu/reject_count_nu)
    return lambda_samples, mu_samples, nu_samples, p_samples, filename     






