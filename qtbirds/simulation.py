import subprocess
import json
import uuid
import os
import numpy as np
from timeit import default_timer as timer
from .utils import load_data
from .linear_algebra import calc_jump_matrix, possible_message_states
from .models import PhenotypicModel, MolecularModel
import treeppl
import treeppl_utils as tu
import pandas as pd
import math as Math

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
    output_filename = f"output-{unique_id}.json"

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

def run_inference(tree, tree_label="No label", prior=None, pa=0.5, pb=0.5, norm_q_mol=None, norm_q_char=None, total_samples=100, sweep_samples=5000, mthd="smc-apf", oss=20, outputf="output.csv"):
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
    
    print("Matrices set up. Attempting to compile...", os.path.join(tppl_path, "models/pheno-mol/qt.tppl"), sweep_samples, mthd)
    with treeppl.Model(filename=os.path.join(tppl_path, "models/pheno-mol/qt.tppl"), samples=sweep_samples, method=mthd) as qtbirds:
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


