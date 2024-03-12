import pandas as pd 
import numpy as np
import qtbirds as qb
import treeppl_utils as tu
import os
from scipy.stats import gamma

qthome, mcore_libs, qt_webppl_home, dep_home = qb.check_os_environment()
script_dir = os.path.dirname(os.path.abspath(__file__))

fasta_file=os.path.join(script_dir, 'data' , 'sample.fasta') # DNA at root of tree
tree_file=os.path.join(script_dir, 'data', 'jeremy-crbd.tre.phyjson') # Actual tree
states=os.path.join(script_dir, 'data' , '2state.json') # Possible states

print(fasta_file, tree_file, states)

mol_model=qb.MolecularModel.JC69
pheno_model=qb.PhenotypicModel.MK_2

N = 1  # Number of simulations
M = 1  # Number of inferences per simulation
particles = 1000
total_samples = 50

# Initialize an empty DataFrame to store results
all_results = pd.DataFrame()

for simulation in range(N):
    # Sample lam, mu, and nu for each simulation
    shape, scale = 3.0, 0.1
    lam = gamma.rvs(a=shape, scale=scale)
    mu = gamma.rvs(a=shape, scale=0.000000001)
    nu = gamma.rvs(a=shape, scale=0.000000001)

    # Generate Q matrices
    norm_q_mol = mol_model.get_q_matrix()
    norm_q_char = pheno_model.get_q_matrix()

    print("Sampled: ", lam, mu, nu);

    # Run the simulation
    output_file = qb.run_simulation( qt_webppl_home=qt_webppl_home
                                   , dep_home=dep_home
                                   , fasta_file=fasta_file
                                   , tree_file=tree_file
                                   , states_file=states
                                   , mol_model=mol_model
                                   , pheno_model=pheno_model
                                   , lam=lam
                                   , mu=mu
                                   , nu=nu
                                   )
    
    # QTBirds inference
    print("Calling inference...")
    
    for inference in range(M):
        # Run inference
        data = qb.load_data(output_file)  # Use the output from the simulation
        tree = data[0]['value']
        tree_label = f"tree_{output_file.split('_')[-1].split('.')[0]}"
        
        prior = {'lam': {'shape': shape, 'scale': scale}, 
                 'mu': {'shape': shape, 'scale': 0.000000001}, 
                 'nu': {'shape': shape, 'scale': 0.000000001}}

        lambda_samples, mu_samples, nu_samples, p_samples, lweights, tree_id = qb.run_inference(
            tree, prior=prior, norm_q_mol=norm_q_mol, norm_q_char=norm_q_char, 
            total_samples=total_samples, sweep_samples=particles, oss=-1)

        # Calculate MAP for lambda, mu, and nu
        #w = np.exp(lweights - np.max(lweights))
        mode_lam = tu.find_MAP(lambda_samples, lweights)
        mean_lam = tu.find_mean(lambda_samples, lweights)
        
        mode_mu = tu.find_MAP(mu_samples, lweights)
        mean_mu = tu.find_mean(mu_samples, lweights)
        
        mode_nu = tu.find_MAP(nu_samples, lweights)
        mean_nu = tu.find_mean(nu_samples, lweights)

        hdpi_lam, hdpi_lam_low, hdpi_lam_high = tu.find_min_hdpi_prob_bin(lam, lambda_samples, lweights)   
        hdpi_mu, hdpi_mu_low, hdpi_mu_high = tu.find_min_hdpi_prob_bin(mu, mu_samples, lweights)
        hdpi_nu, hdpi_nu_low, hdpi_nu_high = tu.find_min_hdpi_prob_bin(nu, nu_samples, lweights)

        lam_95_low, lam_95_high = tu.compute_hdpi(lambda_samples, lweights)
        mu_95_low, mu_95_high = tu.compute_hdpi(mu_samples, lweights)
        nu_95_low, nu_95_high = tu.compute_hdpi(mu_samples, lweights)

        print(hdpi_lam, hdpi_lam_low, hdpi_lam_high);

        # Append results to the DataFrame
        results_df = pd.DataFrame({
            'lam': [lam],
            'mu': [mu],
            'nu': [nu],
            'mode_lam': [mode_lam],
            'mode_mu': [mode_mu],
            'mode_nu': [mode_nu],
            'hdpi_lam': [hdpi_lam],
            'hdpi_lam_low': [hdpi_lam_low],
            'hdpi_lam_high': [hdpi_lam_high],
            'hdpi_mu': [hdpi_mu],
            'hdpi_mu_low': [hdpi_mu_low],
            'hdpi_mu_high': [hdpi_mu_high],
            'hdpi_nu': [hdpi_nu],
            'hdpi_nu_low': [hdpi_nu_low],
            'hdpi_nu_high': [hdpi_nu_high],
            'mean_lam': [mean_lam],
            'mean_mu': [mean_mu],
            'mean_nu': [mean_nu],
            'lam_95_low': [lam_95_low],
            'lam_95_high': [lam_95_high],
            'mu_95_low': [mu_95_low],
            'mu_95_high': [mu_95_high],
            'nu_95_low': [nu_95_low],
            'nu_95_high': [nu_95_high],
            'tree_uuid': [tree_id],
            'simulation': [simulation],
            'inference': [inference]
        })
        
        all_results = pd.concat([all_results, results_df], ignore_index=True)        # Save the results to a CSV file
        file_path = "results.csv"  # Update with your desired path
        all_results.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")