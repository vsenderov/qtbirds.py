import subprocess
import json
import uuid
import os
import numpy as np
import pandas as pd
import math as Math
import treeppl
import treeppl_utils as tu
from timeit import default_timer as timer
from typing import Optional, Dict, Tuple


from .utils import load_data
from .linear_algebra import calc_jump_matrix, possible_message_states
from .models import PhenotypicModel, MolecularModel
from .QTTree import QTNode

def run_simulation( qt_webppl_home
                  , dep_home
                  , fasta_file
                  , tree_file
                  , states_file
                  , mol_model=MolecularModel.JC69
                  , pheno_model=PhenotypicModel.MK_2
                  , lam=0.1
                  , mu=0.1
                  , nu=0.0
                  ):
    """
    This function does a Q-T-Birds simulation by invoking the WebPPL simulator
    via a subprocess command.

    Side-effects: two files, one with rates, and another with the simulation.

    TODO: Modify the function to allow supplying Q-matrices directly from Python, 
    instead of relying on pregenerated files.

    :param qt_webppl_home: The directory where the simulator is installed
    :param dep_home: The directory, in which the dependency packages are subdirs of
    :param fasta_file: The sequence at the root of the tree (default provided)
    :param tree_file: The PhyJSON tree to be enriched (default provided)
    :param states_file: The phenotypic states (default provided)
    :param mol_model: The nuclear transitions matrix (default provided)
    :param pheno_model: The phenotypic transitions matrix (default provided)
    :param lam: Phenotypic rate
    :param mu: Molecular rate
    :param nu: Joint rate
    
    :return: The filename with the enriched tree.
    """
    # Generate a unique identifier
    unique_id = uuid.uuid4()

    # Generate rates JSON
    rates = {
        "lam": lam,
        "mu": mu,
        "nu": nu
    }
    rates_filename = f"rates-{unique_id}.json"
    with open(rates_filename, 'w') as file:
        json.dump(rates, file)

    # Generate unique output filename
    output_filename = f"tree_{unique_id}.json"

    # Construct the command using os.path.join for robust path handling
    # command = f"webppl {os.path.join(qt_webppl_home, 'qtbirds-sim.wppl')} --require {os.path.join(dep_home, 'fasta2json')} " \
    #           f"--require {qt_webppl_home} --require webppl-fs --require {os.path.join(dep_home, 'phywppl/phyjs')} -- " \
    #           f"{os.path.join(qt_webppl_home, fasta_file)} {os.path.join(qt_webppl_home, tree_file)} {rates_filename} " \
    #           f"{os.path.join(qt_webppl_home, states)} {os.path.join(qt_webppl_home, nucleo_json)} " \
    #           f"{os.path.join(qt_webppl_home, pheno_json)} 1 > {output_filename}"
    
    command = f"webppl {os.path.join(qt_webppl_home, 'qtbirds-sim.wppl')} --require {os.path.join(dep_home, 'fasta2json')} " \
              f"--require {qt_webppl_home} --require webppl-fs --require {os.path.join(dep_home, 'phywppl/phyjs')} -- " \
              f"{fasta_file} {tree_file} {rates_filename} " \
              f"{states_file} {os.path.join(qt_webppl_home, mol_model.get_filename())} " \
              f"{os.path.join(qt_webppl_home, pheno_model.get_filename())} 1 > {output_filename}"

    

    print(command)
    # Execute the command
    subprocess.run(command, shell=True)

    # Clean up the rates file
    # os.remove(rates_filename)  # Uncomment this line if you want to delete the rates file after execution

    print(f"Simulation output saved in {output_filename}")
    return output_filename

def run_inference(tree, tree_label="No label", prior=None, pa=0.5, pb=0.5, norm_q_mol=None, norm_q_char=None, total_samples=100, sweep_samples=5000, mthd="smc-apf", oss=20, outputf="output.csv", custominf=None):
    """
    Run inference on a given tree data file with specified prior distribution and Q-matrices.

    You might need to e.g. from bash do:
    
    export QTHOME=/home/viktor/Sync/Workspaces/Q-T-Birds/
    export MCORE_LIBS="$MCORE_LIBS:treeppl=$HOME/.local/src/treeppl/"

    :param tree: The actual tree
    :param tree_label: The label (used to be derived from the filename)
    :param prior: A dictionary specifying the prior distribution for lambda, mu, and nu (default values provided)
    :param p: prior probability of corelation
    :param norm_q_mol: The Q-matrix for molecular data (default Jukes-Cantor matrix provided)
    :param norm_q_char: The Q-matrix for character data (default Markov k=2 matrix provided)
    :param total_samples: Total number of samples to generate (default 100)
    :param sweep_samples: Number of samples per sweep, particles (default 5000)
    :param outputf: File to store intermediate results 
    :return: A tuple containing samples of lambda, mu, nu, the weights, and the tree identifier
    """
    print("Running inference...")

    if custominf == None:
        custominf = os.path.join(tppl_path, "models/pheno-mol/qt.tppl")

    # Environment extraction
    qthome = os.environ.get('QTHOME')
    if not qthome:
        raise ValueError("The QTHOME environment variable is not set.")
    
    # Set default values for the prior and Q-matrices if not provided
    if prior is None:
        prior = {'lam': {'shape': 1.0, 'scale': 0.5}, 'mu': {'shape': 1.0, 'scale': 0.5}, 'nu': {'shape': 1.0, 'scale': 0.5}}
    if norm_q_mol is None:
        norm_q_mol = np.array([[-1., 1/3, 1/3, 1/3], [1/3, -1., 1/3, 1/3], [1/3, 1/3, -1., 1/3], [1/3, 1/3, 1/3, -1.]])
    if norm_q_char is None:
        norm_q_char = np.array([[-1., 1.], [1., -1.]])

    # Calculate jump matrices
    jMol = calc_jump_matrix(norm_q_mol)
    jChar = calc_jump_matrix(norm_q_char)

    # Define startMessages based on the shape of norm_q_char
    startMessages = possible_message_states(norm_q_char.shape[0])

    # Load the data
    # WIP
    #data = load_data(tree_data)
    #tree = data[0]['value']
    #tree_label = f"tree_{tree_data.split('_')[-1].split('.')[0]}"
   

    # Initialize lists to store samples
    lambda_samples = []
    mu_samples = []
    nu_samples = []
    p_samples = []
    lweights = []

    # Run the model
    tppl_src = os.environ.get('MCORE_LIBS')
    # Extracting the part after "treeppl="
    tppl_path = tppl_src.split('treeppl=')[-1] if 'treeppl=' in tppl_src else None
    
    print("Matrices set up. Attempting to compile...", custominf, sweep_samples, mthd)
    with treeppl.Model(filename=custominf, samples=sweep_samples, method=mthd) as qtbirds:
        print("Model compiled. Running inference with", sweep_samples, "samples/particles and", mthd);
        start = timer()
        if (oss < 1):
            res = qtbirds(tree=tree, normQChar=norm_q_char, jChar=jChar, charMessages=startMessages,
                          normQMol=norm_q_mol, jMol=jMol,
                          lamShape=prior['lam']['shape'], lamScale=prior['lam']['scale'],
                          muShape=prior['mu']['shape'], muScale=prior['mu']['scale'],
                          nuShape=prior['nu']['shape'], nuScale=prior['nu']['scale'],
                          pa=pa, pb=pb)
            oss = Math.ceil(tu.ess(res))
        else:
            oss = oss
        end = timer()
        print("Exploratory sweep completed. OSS = ", oss)
        print("Seconds per sample: ", (end - start)/oss)
        while len(lambda_samples) < total_samples:
            res = qtbirds(tree=tree, normQChar=norm_q_char, jChar=jChar, charMessages=startMessages,
                          normQMol=norm_q_mol, jMol=jMol,
                          lamShape=prior['lam']['shape'], lamScale=prior['lam']['scale'],
                          muShape=prior['mu']['shape'], muScale=prior['mu']['scale'],
                          nuShape=prior['nu']['shape'], nuScale=prior['nu']['scale'],
                          pa=pa, pb=pb)
            # Extract samples and log weights
            subsamples = res.subsample(oss)
            #print("Subsamples structure:", subsamples)  # Add this line for debugging

            for sample in subsamples:
                lambda_samples.append(sample[0])  # Extract lambda
                mu_samples.append(sample[1])      # Extract mu
                nu_samples.append(sample[2])      # Extract nu
                p_samples.append(sample[3])     # Extract rho

            lweights.extend([res.norm_const] * oss)
            
            import pandas as pd

            # Assuming the initialization of subsamples, lambda_samples, mu_samples, nu_samples, p_samples, and lweights is done before this snippet.

            # Create a data frame with the specified columns
            data_frame = pd.DataFrame({
                'lambda_samples': lambda_samples,
                'mu_samples': mu_samples,
                'nu_samples': nu_samples,
                'p_samples': p_samples,
                'lweights': lweights
            })

            # Write the data frame to the file named outputf, overwrite if it exists
            data_frame.to_csv(outputf, index=False)  # Write to CSV without the index column
            
            print(f"So far {len(lambda_samples)} samples; present run log Z = {res.norm_const}; total var log Z = {np.var(lweights)}")

    return lambda_samples, mu_samples, nu_samples, p_samples, lweights, tree_label




def run_inference_multithreaded(
    tree: QTNode,
    label: str = "No label",
    prior: Optional[Dict[str, Dict[str, float]]] = None,
    norm_q_mol: Optional[np.ndarray] = None,
    norm_q_char: Optional[np.ndarray] = None,
    total_samples: int = 100,
    particles: int = 5000,
    mthd: str = "smc-apf",
    sweep_samples: int = 1,
    numthreads: int = 6,
    outputf: str = "output-inference.csv",
    qt: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    
    """
    Run inference on a given tree data file with specified prior distribution and Q-matrices.
    
    :param prior: The hyper-parameters of the Gamma distribution for λ, μ, ν (Gamma) 
    and p (Beta) e.g.

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
        
    TODO mthd should be an Enum
        
    :return: Samples from λ, μ, ν, p and the label of the tree
    """
    
    print("Running multithreaded inference with " + str(numthreads) + " threads...")
    
    if prior is None:
        print("default prior")
        prior = {
            'lam': {'shape': 1.0, 'scale': 0.5},
            'mu': {'shape': 1.0, 'scale': 0.5},
            'nu': {'shape': 1.0, 'scale': 0.5},
            'p': {'pa': 0.5, 'pb': 0.5}
            }
        
    if norm_q_mol is None:
        print("default molecular model: JK")
        norm_q_mol = np.array( [[-1., 1/3, 1/3, 1/3],
                                [1/3, -1., 1/3, 1/3],
                                [1/3, 1/3, -1., 1/3],
                                [1/3, 1/3, 1/3, -1.]])
        
    if norm_q_char is None:
        print("default phenotype model: binary Mk")
        norm_q_char = np.array([[-1., 1.],
                                [1., -1.]])

    jMol = calc_jump_matrix(norm_q_mol)
    jChar = calc_jump_matrix(norm_q_char)
    startMessages = possible_message_states(norm_q_char.shape[0])

    lambda_samples = []
    mu_samples = []
    nu_samples = []
    p_samples = []
    lweights = []

    tppl_src = os.environ.get('MCORE_LIBS') # Extracting the part after "treeppl="
    tppl_path = tppl_src.split('treeppl=')[-1] if 'treeppl=' in tppl_src else None
    if qt is None:
        qt = os.path.join(tppl_path, "models/pheno-mol/qt.tppl")
    
    print("Matrices set up. Attempting to compile: ", qt, particles, mthd)
    
    # TODO -n <subsample size> (single minus)
    with treeppl.Model(filename=qt, samples=particles, method=mthd, subsample=True) as qtbirds:
        print("Model compiled. Running inference with", particles, "samples/particles and", mthd);

        if (sweep_samples < 1):
            print("Exploratory sweep.")
            start = timer()
            res = qtbirds(  tree=tree, normQChar=norm_q_char, jChar=jChar, charMessages=startMessages,
                            normQMol=norm_q_mol, jMol=jMol,
                            lamShape=prior['lam']['shape'], lamScale=prior['lam']['scale'],
                            muShape=prior['mu']['shape'], muScale=prior['mu']['scale'],
                            nuShape=prior['nu']['shape'], nuScale=prior['nu']['scale'],
                            pa=prior['p']['pa'],
                            pb=prior['p']['pb']
                            )
            sweep_samples = Math.ceil(tu.ess(res))
            end = timer()
            print("Exploratory sweep completed. OSS = ", sweep_samples)
            print("Seconds per sample: ", (end - start)/sweep_samples)
        else:
            sweep_samples = sweep_samples
        
        while len(lambda_samples) < total_samples:
            #TODO https://chat.openai.com/share/6745c36f-591f-4629-b961-eddd098bd19a
            
            threads = []
            results = [None] * numthreads
            
            for _ in range(numthreads):
                thread = tu.ThreadWithReturnValue(  target=qtbirds,
                                                    kwargs={
                                                        'tree': tree,
                                                        'normQChar': norm_q_char,
                                                        'jChar': jChar,
                                                        'charMessages': startMessages,
                                                        'normQMol': norm_q_mol,
                                                        'jMol': jMol,
                                                        'lamShape': prior['lam']['shape'],
                                                        'lamScale': prior['lam']['scale'],
                                                        'muShape': prior['mu']['shape'],
                                                        'muScale': prior['mu']['scale'],
                                                        'nuShape': prior['nu']['shape'],
                                                        'nuScale': prior['nu']['scale'],
                                                        'pa': prior['p']['pa'],
                                                        'pb': prior['p']['pb'] }
                                                    )
                threads.append(thread)
                thread.start()
            
            for i, thread in enumerate(threads):
                if i < numthreads:
                    results[i] = thread.join()  # Wait for the thread to complete and get the return value
                    #os.sleep(200)
                    subsamples = (results[i]).getsample()
                    #subsamples = (results[i]).subsample(1)
                    for sample in subsamples:
                        lambda_samples.append(sample[0])  # Extract lambda
                        mu_samples.append(sample[1])      # Extract mu
                        nu_samples.append(sample[2])      # Extract nu
                        p_samples.append(sample[3])     # Extract rho
                        
                    lweights.extend([results[i].norm_const] * sweep_samples)
                        
                        #print(mu_samples)
                        #print(nu_samples)
                        #print(p_samples)
                        #print(lweights)

                else:
                    thread.join()  # Make sure to join remaining threads even if their results are not collected
            
            data_frame = pd.DataFrame({
                'lambda_samples': lambda_samples,
                'mu_samples': mu_samples,
                'nu_samples': nu_samples,
                'p_samples': p_samples,
                'lweights': lweights
            })
            data_frame.to_csv("output_" + label + ".csv", index=False)
            
            print(f"So far {len(lambda_samples)} samples; var log Z = {np.var(lweights)}")

    return lambda_samples, mu_samples, nu_samples, p_samples, lweights, label


###
# MCMC Inference with TreePPL
###

def run_mcmc_inference  ( tree: QTNode
                        , label: str = "no-label"
                        , prior: Optional[Dict[str, Dict[str, float]]] = None
                        , norm_q_mol: Optional[np.ndarray] = None
                        , norm_q_char: Optional[np.ndarray] = None
                        , samples: int = 100
                        , burnin: int = 5000
                        , thinning: int = 1000
                        , chains: int = None
                        , custominf: int = None
                        , cps: str = "partial"
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    print("TreePPL MCMC inference...")
    
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

    # Environment extraction
    qthome = os.environ.get('QTHOME')
    if not qthome:
        raise ValueError("The QTHOME environment variable is not set.")

    # Calculate jump matrices
    jMol = calc_jump_matrix(norm_q_mol)
    jChar = calc_jump_matrix(norm_q_char)

    # Define startMessages based on the shape of norm_q_char
    startMessages = possible_message_states(norm_q_char.shape[0])
   

    # Initialize lists to store samples
    lambda_samples = []
    mu_samples = []
    nu_samples = []
    p_samples = []
    lweights = []

    # Run the model
    tppl_src = os.environ.get('MCORE_LIBS')
    # Extracting the part after "treeppl="
    tppl_path = tppl_src.split('treeppl=')[-1] if 'treeppl=' in tppl_src else None
    
    if custominf == None:
        custominf = os.path.join(tppl_path, "models/pheno-mol/qt.tppl")
    
    print("Matrices set up. Attempting to compile...", custominf)

    total = samples*thinning + burnin

    
    with treeppl.Model(filename=custominf, method="mcmc-lightweight", align=True, samples=total) as qtbirds:
        print   ( "TreePPL MCMC model compiled. Running inference for a total of "
                , total
                , "samples, including burn-in and thinning"
                )

        res = qtbirds(tree=tree, normQChar=norm_q_char, jChar=jChar, charMessages=startMessages,
                          normQMol=norm_q_mol, jMol=jMol,
                          lamShape=prior['lam']['shape'], lamScale=prior['lam']['scale'],
                          muShape=prior['mu']['shape'], muScale=prior['mu']['scale'],
                          nuShape=prior['nu']['shape'], nuScale=prior['nu']['scale'],
                          pa=prior['p']['pa'], pb=prior['p']['pb'])
        


        # Extracting values
        lambda_samples = [sample[0] for sample in res.samples]
        mu_samples = [sample[1] for sample in res.samples]
        nu_samples = [sample[2] for sample in res.samples]
        p_samples = [sample[3] for sample in res.samples]
        weights = res.weights  #
        
        import pandas as pd

        # Assuming the initialization of subsamples, lambda_samples, mu_samples, nu_samples, p_samples, and lweights is done before this snippet.

        # Create a data frame with the specified columns
        data_frame = pd.DataFrame({
            'lambda_samples': lambda_samples,
            'mu_samples': mu_samples,
            'nu_samples': nu_samples,
            'p_samples': p_samples,
            'lweights': weights
        })[burnin::thinning]

        # Write the data frame to the file named outputf, overwrite if it exists
        data_frame.to_csv("mcmc_tppl_output_" + label + ".csv", index=False)

    return lambda_samples, mu_samples, nu_samples, p_samples, label



###
# MCMC Inference with TreePPL
###

def run_mcmc_inference_multithreaded  ( tree: QTNode
                        , label: str = "no-label"
                        , prior: Optional[Dict[str, Dict[str, float]]] = None
                        , norm_q_mol: Optional[np.ndarray] = None
                        , norm_q_char: Optional[np.ndarray] = None
                        , samples: int = 100
                        , burnin: int = 0
                        , thinning: int = 1
                        , chains: int = 1
                        , custominf: int = None
                        , gprob:float = 0.1
                        , drift:float = 0.01
                        , cps:str = "partial"
                        #) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
                        ) -> str:
    print("TreePPL MCMC inference...")
    
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

    # Environment extraction
    qthome = os.environ.get('QTHOME')
    if not qthome:
        raise ValueError("The QTHOME environment variable is not set.")

    # Calculate jump matrices
    jMol = calc_jump_matrix(norm_q_mol)
    jChar = calc_jump_matrix(norm_q_char)

    # Define startMessages based on the shape of norm_q_char
    startMessages = possible_message_states(norm_q_char.shape[0])
   

    # Initialize lists to store samples
    all_lambda_samples = []
    all_mu_samples = []
    all_nu_samples = []
    all_p_samples = []
    all_weights = []

    # Run the model
    tppl_src = os.environ.get('MCORE_LIBS')
    # Extracting the part after "treeppl="
    tppl_path = tppl_src.split('treeppl=')[-1] if 'treeppl=' in tppl_src else None
    
    if custominf == None:
        custominf = os.path.join(tppl_path, "models/pheno-mol/qt.tppl")
    
    print("Matrices set up. Attempting to compile...", custominf)

    total = samples*thinning + burnin


    # --mcmc-lw-gprob <value>            The probability of performing a global MH 
    #                                 step (non-global means only modify a single
    #                                 sample in the previous trace). Default: 
    #                                 0.1.
    #
    
    with treeppl.Model(filename=custominf, method="mcmc-lightweight", align=True, cps=cps, drift=drift, samples=total, mcmc_lw_gprob=gprob) as qtbirds:
    #with treeppl.Model(filename=custominf, method="mcmc-lightweight", align=True, cps='full', samples=total, mcmc_lw_gprob=gprob) as qtbirds:
        print   ( "TreePPL MCMC model compiled. Running multithread MCMC inference for a total of"
                , total
                , "samples, including burn-in and thinning"
                )

        threads = []
        results = [None] * chains
        
        for _ in range(chains):
            thread = tu.ThreadWithReturnValue(
                target=qtbirds,
                kwargs={
                    'tree': tree,
                    'normQChar': norm_q_char,
                    'jChar': jChar,
                    'charMessages': startMessages,
                    'normQMol': norm_q_mol,
                    'jMol': jMol,
                    'lamShape': prior['lam']['shape'],
                    'lamScale': prior['lam']['scale'],
                    'muShape': prior['mu']['shape'],
                    'muScale': prior['mu']['scale'],
                    'nuShape': prior['nu']['shape'],
                    'nuScale': prior['nu']['scale'],
                    'pa': prior['p']['pa'],
                    'pb': prior['p']['pb']
                    }
                )
            
            threads.append(thread)
            thread.start()
        

        for i, thread in enumerate(threads):
            if i < chains:
                results[i] = thread.join()  # Wait for the thread to complete and get the return value        

                # Extracting values
                lambda_samples = [sample[0] for sample in results[i].samples]
                mu_samples = [sample[1] for sample in results[i].samples]
                nu_samples = [sample[2] for sample in results[i].samples]
                p_samples = [sample[3] for sample in results[i].samples]
                weights = results[i].weights  #
                
                # Apply burnin and thinning
                lambda_samples = lambda_samples[burnin::thinning]
                mu_samples = mu_samples[burnin::thinning]
                nu_samples = nu_samples[burnin::thinning]
                p_samples = p_samples[burnin::thinning]
                weights = weights[burnin::thinning]
                
                # Append to the global lists
                all_lambda_samples.extend(lambda_samples)
                all_mu_samples.extend(mu_samples)
                all_nu_samples.extend(nu_samples)
                all_p_samples.extend(p_samples)
                all_weights.extend(weights)
                
                # Create a data frame with the specified columns
                import pandas as pd
                data_frame = pd.DataFrame({
                    'lambda_samples': lambda_samples,
                    'mu_samples': mu_samples,
                    'nu_samples': nu_samples,
                    'p_samples': p_samples,
                    'lweights': weights
                })

                # Write the data frame to the file named outputf, overwrite if it exists
                data_frame.to_csv("mcmc_tppl_output_" + label + '-' + str(i) + ".csv", index=False)
                
            else:
                thread.join()  # Make sure to join remaining threads even if their results are not collected
        
    # Convert lists to numpy arrays
    final_lambda_samples = np.array(all_lambda_samples)
    final_mu_samples = np.array(all_mu_samples)
    final_nu_samples = np.array(all_nu_samples)
    final_p_samples = np.array(all_p_samples)
    final_weights = np.array(all_weights)

    return final_lambda_samples, final_mu_samples, final_nu_samples, final_p_samples, label
