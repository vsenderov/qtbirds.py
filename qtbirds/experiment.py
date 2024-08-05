import os
import time
import shutil
import pandas as pd 
import numpy as np
import qtbirds as qb
import treeppl_utils as tu
from scipy.stats import gamma

# no need to import typing for basic types
def run_smc_experiment_real_data( nexus_file: str
                                , fasta_file: str
                                , csv_file: str
                                , mol_model: str
                                , pheno_model: str
                                , exp_name: str
                                , shape: float
                                , scale: float
                                , pa: float
                                , pb: float
                                , samples: int
                                , M: int # number of inferences per simulation (replicates)
                                , particles: int
                                , numthreads: int
                                , method: str
                                , ppl: str):
    print("Running", exp_name)
    qthome, mcore_libs, qt_webppl_home, dep_home = qb.check_os_environment()
    script_dir = os.getcwd()
    nexus_file = os.path.join(script_dir, 'data', nexus_file)
    fasta_file = os.path.join(script_dir, 'data', fasta_file)
    csv_file = os.path.join(script_dir, 'data', csv_file)
    mol_model = qb.MolecularModel.JC69
    pheno_model = qb.PhenotypicModel.MK_2
    tree = qb.read_external_data(nexus_file, fasta_file, csv_file)
    prior = {
        'lam': {'shape': shape, 'scale': scale}, 
        'mu': {'shape': shape, 'scale': scale}, 
        'nu': {'shape': shape, 'scale': scale},
        'p': {'pa': pa, 'pb': pb}
        }
    norm_q_mol = mol_model.get_q_matrix()
    norm_q_char = pheno_model.get_q_matrix()
    all_results = pd.DataFrame()
    file_path = exp_name + "-" + str(samples) + "-results.csv" 
    print(f"Data will be saved to {file_path}")
    output_directory = f"{exp_name}-output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    tree_label, _ = os.path.splitext(os.path.basename(nexus_file))
    tree_label = tree_label + '-' + str(samples)
    
    for inference in range(M):
        print("SMC inference (multithreaded) on ", tree_label, " ...")
        
        start_time = time.time()
        lambda_samples, mu_samples, nu_samples, p_samples, lweights, tree_id = qb.run_inference_multithreaded(
                tree,
                label=tree_label, 
                prior=prior,
                norm_q_mol=norm_q_mol,
                norm_q_char=norm_q_char,
                total_samples=samples,
                particles=particles,
                mthd=method,
                numthreads=numthreads,
                sweep_samples=1, 
                qt=ppl)                       
        elapsed_time = time.time() - start_time
        
        print("SMC Time for replicate " + str(inference) + ": " + str(elapsed_time))

        # Calculate point estimates for λ, μ, ν
        mode_lam = tu.find_MAP(lambda_samples, lweights)
        mean_lam = tu.find_mean(lambda_samples, lweights)
        mode_mu = tu.find_MAP(mu_samples, lweights)
        mean_mu = tu.find_mean(mu_samples, lweights)
        mode_nu = tu.find_MAP(nu_samples, lweights)
        mean_nu = tu.find_mean(nu_samples, lweights)

        hdpi_lam, hdpi_lam_low, hdpi_lam_high = tu.find_min_hdpi_prob(lam, lambda_samples, lweights)   
        hdpi_mu, hdpi_mu_low, hdpi_mu_high = tu.find_min_hdpi_prob(mu, mu_samples, lweights)
        hdpi_nu, hdpi_nu_low, hdpi_nu_high = tu.find_min_hdpi_prob(nu, nu_samples, lweights) 

        # For reference add 95% HDPI intervals (default)
        lam_95_low, lam_95_high = tu.compute_hdpi(lambda_samples)
        mu_95_low, mu_95_high = tu.compute_hdpi(mu_samples)
        nu_95_low, nu_95_high = tu.compute_hdpi(mu_samples)

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
            'inference': [inference],
            'type': ['smc-tppl'],
            'elapsed_time': [elapsed_time]
        })

        all_results = pd.concat([all_results, results_df], ignore_index=True)        # Save the results to a CSV file
        all_results.to_csv(file_path, index=False)

        for file_name in os.listdir('.'):
            # Check if the file contains the string in tree_label
            if tree_label in file_name:
                # Construct full paths
                source_path = os.path.join('.', file_name)
                destination_path = os.path.join(output_directory, file_name)
        
                # Move file to the output directory
                shutil.move(source_path, destination_path)
                print(f"Moved {file_name} to {output_directory}")
                
                
                
def run_mcmc_experiment_real_data( nexus_file: str
                                , fasta_file: str
                                , csv_file: str
                                , mol_model: str
                                , pheno_model: str
                                , exp_name: str
                                , shape: float
                                , scale: float
                                , pa: float
                                , pb: float
                                , samples: int
                                , M: int # number of inferences per simulation (replicates)
                                , burnin: int
                                , thinning: int
                                , drift: float
                                , gprob: float
                                , numthreads: int # chain
                                , ppl: str
                                , cps: str):
    # PRINT
    print("Running", exp_name)
    
    # VAR SETUP
    qthome, mcore_libs, qt_webppl_home, dep_home = qb.check_os_environment()
    script_dir = os.getcwd()
    nexus_file = os.path.join(script_dir, 'data', nexus_file)
    fasta_file = os.path.join(script_dir, 'data', fasta_file)
    csv_file = os.path.join(script_dir, 'data', csv_file)
    mol_model = qb.MolecularModel.JC69
    pheno_model = qb.PhenotypicModel.MK_2
    tree = qb.read_external_data(nexus_file, fasta_file, csv_file)
    prior = {
        'lam': {'shape': shape, 'scale': scale}, 
        'mu': {'shape': shape, 'scale': scale}, 
        'nu': {'shape': shape, 'scale': scale},
        'p': {'pa': pa, 'pb': pb}
        }
    norm_q_mol = mol_model.get_q_matrix()
    norm_q_char = pheno_model.get_q_matrix()
    
    # OUTPUT SETUP
    all_results = pd.DataFrame()
    file_path = exp_name + "-" + str(samples) + "-results.csv" 
    print(f"Data will be saved to {file_path}")
    output_directory = f"{exp_name}-output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    tree_label, _ = os.path.splitext(os.path.basename(nexus_file))
    tree_label = tree_label + '-' + str(samples)
    
    # LOOP
    for inference in range(M):
        print("MCMC inference (multithreaded) on ", tree_label, " ...")
        
        start_time = time.time()
        lambda_samples, mu_samples, nu_samples, p_samples, lweights, tree_id = qb.run_mcmc_inference_multithreaded(
                tree=tree,
                label=tree_label, 
                prior=prior,
                norm_q_mol=norm_q_mol,
                norm_q_char=norm_q_char,
                samples=samples,
                burnin=burnin,
                thinning=thinning,
                drift=drift,
                chains=numthreads,
                gprob=gprob,
                custominf=ppl,
                cps=cps)                       
        elapsed_time = time.time() - start_time
        
        print("SMC Time for replicate " + str(inference) + ": " + str(elapsed_time))

        # Calculate point estimates for λ, μ, ν
        mode_lam = tu.find_MAP(lambda_samples, lweights)
        mean_lam = tu.find_mean(lambda_samples, lweights)
        mode_mu = tu.find_MAP(mu_samples, lweights)
        mean_mu = tu.find_mean(mu_samples, lweights)
        mode_nu = tu.find_MAP(nu_samples, lweights)
        mean_nu = tu.find_mean(nu_samples, lweights)

        hdpi_lam, hdpi_lam_low, hdpi_lam_high = tu.find_min_hdpi_prob(lam, lambda_samples, lweights)   
        hdpi_mu, hdpi_mu_low, hdpi_mu_high = tu.find_min_hdpi_prob(mu, mu_samples, lweights)
        hdpi_nu, hdpi_nu_low, hdpi_nu_high = tu.find_min_hdpi_prob(nu, nu_samples, lweights) 

        # For reference add 95% HDPI intervals (default)
        lam_95_low, lam_95_high = tu.compute_hdpi(lambda_samples)
        mu_95_low, mu_95_high = tu.compute_hdpi(mu_samples)
        nu_95_low, nu_95_high = tu.compute_hdpi(mu_samples)

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
            'inference': [inference],
            'type': ['smc-tppl'],
            'elapsed_time': [elapsed_time]
        })

        all_results = pd.concat([all_results, results_df], ignore_index=True)        # Save the results to a CSV file
        all_results.to_csv(file_path, index=False)

        for file_name in os.listdir('.'):
            # Check if the file contains the string in tree_label
            if tree_label in file_name:
                # Construct full paths
                source_path = os.path.join('.', file_name)
                destination_path = os.path.join(output_directory, file_name)
        
                # Move file to the output directory
                shutil.move(source_path, destination_path)
                print(f"Moved {file_name} to {output_directory}")