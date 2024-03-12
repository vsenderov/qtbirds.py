
import qtbirds as qb
import os
qthome, mcore_libs, qt_webppl_home, dep_home = qb.check_os_environment()
script_dir = os.path.dirname(os.path.abspath(__file__))

# CONFIGURATION
nexus_file = os.path.join(qthome, 'data', 'Mammals_tree_Vertlife', 'Tiny_mammal_tree_Vertlife', 'output.nex')
fasta_file = os.path.join(qthome, 'pelican', 'data', 'orthomam', 'alignments', 'ENSG00000170615_SLC26A5_NT.fasta')
csv_file = os.path.join(qthome, 'pelican', 'data', 'orthomam', 'phenotypes', 'echolocation.tsv')

mol_model=qb.MolecularModel.JC69
pheno_model=qb.PhenotypicModel.MK_2

particles = 100
total_samples = 10

shape, scale = 3.0, 0.1
import sys

# Make sure there's at least one command line argument provided
if len(sys.argv) > 1:
    output_file = sys.argv[1]
    print(output_file)
else:
    print("No command line argument provided for output file.")
    output_file = None  # Or handle the absence of the argument as needed

# END OF CONFIGURATION

data = qb.load_data(output_file)  # Use the output from the simulation
tree = data[0]['value']
tree_label = f"tree_{output_file.split('_')[-1].split('.')[0]}"

prior = {'lam': {'shape': shape, 'scale': scale}, 
        'mu': {'shape': shape, 'scale': scale}, 
        'nu': {'shape': shape, 'scale': 0.000000001}}

norm_q_mol = mol_model.get_q_matrix()
norm_q_char = pheno_model.get_q_matrix()

lambda_samples, mu_samples, nu_samples, p_samples, lweights, tree_id = qb.run_inference(
            tree, prior=prior, norm_q_mol=norm_q_mol, norm_q_char=norm_q_char, 
            total_samples=total_samples, sweep_samples=particles, oss=-1, mthd="smc-bpf")
# import pandas as pd
# df = pd.read_csv(csv_file, sep='\t')
# random_rows1 = df[df['trait'] == 0].sample(n=5)
# random_rows2 = df[df['trait'] == 1].sample(n=5)

# print(random_rows1['species'], random_rows2['species'])

# Now `data` is ready to be processed in the format you described for further analysis
# import json
# # Specify the file path for your JSON file
# json_file_path = 'data.json'

# # Directly use the `to_json` method if `data` is a single instance
# json_str = data.to_json()
# with open(json_file_path, 'w') as json_file:
#     json_file.write(json_str)
