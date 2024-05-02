import qtbirds as qb
import pandas as pd
"""
This example runs the Python-native MCMC inference on a Q-T data file.

A Q-T data file is a phylogenetic file in a JSON format (but it is not PhyJSON!),
which contains tip data.  The tip data is sequence data and phenotype data.

Syntax:

python3 examples/mcmc.py DATA

Example:

python3 examples/mcmc.py examples/data/crbd-1n-1p-44t.json
"""
###
# Options
###

shape, scale = 3.0, 0.1
mol_model=qb.MolecularModel.JC69
pheno_model=qb.PhenotypicModel.MK_2
samples = 10000
burnin = 1000
thinning = 1
chains = 1

###
# Initialization
###
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))

prior = {
    'lam': {'shape': shape, 'scale': scale}, 
    'mu': {'shape': shape, 'scale': scale}, 
    'nu': {'shape': shape, 'scale': scale}
}


# Make sure there's at least one command line argument provided
if len(sys.argv) > 1:
    output_file = sys.argv[1]
    print(output_file)
else:
    print("No command line argument provided for output file.")
    sys.exit(0)

data = qb.load_data(output_file)  # Use the output from the simulation
tree = data[0]['value']
label = f"tree_{output_file.split('_')[-1].split('.')[0]}"

norm_q_mol = mol_model.get_q_matrix()
norm_q_char = pheno_model.get_q_matrix()

lambda_samples, mu_samples, nu_samples, p_samples, filename = qb.qt_mcmc(tree=tree,
            #label=label,
            prior=prior,
            norm_q_mol=norm_q_mol,
            norm_q_char=norm_q_char,
            samples=samples,
            burnin=burnin,
            thinning=thinning)

# Create a DataFrame
df = pd.DataFrame({
    'Lambda': lambda_samples,
    'Mu': mu_samples,
    'Nu': nu_samples
})

print(df)


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame (df) is defined
df.plot(kind='line', subplots=True)  # Plot traces for each parameter
plt.savefig("mcmc_traces.png")  # Save the plot as an image



#lambda_samples, mu_samples, nu_samples, p_samples, lweights, tree_id = qb.run_inference(
#            tree, prior=prior, norm_q_mol=norm_q_mol, norm_q_char=norm_q_char, 
#            total_samples=total_samples, sweep_samples=particles, oss=-1, mthd="smc-bpf")
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
