import subprocess
import json
import uuid
import os
import numpy as np
from timeit import default_timer as timer
from .utils import load_data, optimal_subsample_size
from .linear_algebra import calc_jump_matrix, possible_message_states
import treeppl
import pandas as pd

def run_simulation(qt_webppl_home, dep_home, fasta_file="phylo/sample2.fasta", 
                   tree_file="phylo/jeremy-crbd.tre.phyjson", 
                   states="pheno/2state.json", nucleo_json="nucleo/jc69.Q.json", 
                   pheno_json="pheno/mk-2state.Q.json", lam=0.1, mu=0.1, nu=0.0):
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
    :param states: The phenotypic states (default provided)
    :param nucleo_json: The nuclear transitions matrix (default provided)
    :param mk_json: The phenotypic transitions matrix (default provided)
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
    command = f"webppl {os.path.join(qt_webppl_home, 'qtbirds-sim.wppl')} --require {os.path.join(dep_home, 'fasta2json')} " \
              f"--require {qt_webppl_home} --require webppl-fs --require {os.path.join(dep_home, 'phywppl/phyjs')} -- " \
              f"{os.path.join(qt_webppl_home, fasta_file)} {os.path.join(qt_webppl_home, tree_file)} {rates_filename} " \
              f"{os.path.join(qt_webppl_home, states)} {os.path.join(qt_webppl_home, nucleo_json)} " \
              f"{os.path.join(qt_webppl_home, pheno_json)} 1 > {output_filename}"

    print(command)
    # Execute the command
    subprocess.run(command, shell=True)

    # Clean up the rates file
    # os.remove(rates_filename)  # Uncomment this line if you want to delete the rates file after execution

    print(f"Simulation output saved in {output_filename}")
    return output_filename

def run_inference(tree_data, prior=None, norm_q_mol=None, norm_q_char=None, total_samples=100, sweep_samples=5000, mthd="smc-apf"):
    """
    Run inference on a given tree data file with specified prior distribution and Q-matrices.

    You might need to e.g. from bash do:
    
    export QTHOME=/home/viktor/Sync/Workspaces/Q-T-Birds/
    export MCORE_LIBS="$MCORE_LIBS:treeppl=$HOME/.local/src/treeppl/"

    :param tree_data: The filename of the tree data
    :param prior: A dictionary specifying the prior distribution for lambda, mu, and nu (default values provided)
    :param norm_q_mol: The Q-matrix for molecular data (default Jukes-Cantor matrix provided)
    :param norm_q_char: The Q-matrix for character data (default Markov k=2 matrix provided)
    :param total_samples: Total number of samples to generate (default 100)
    :param sweep_samples: Number of samples per sweep, particles (default 5000) 
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
    data = load_data(tree_data)
    tree = data[0]['value']
    tree_label = f"tree_{tree_data.split('_')[-1].split('.')[0]}"

    # Initialize lists to store samples
    lambda_samples = []
    mu_samples = []
    nu_samples = []
    lweights = []

    # Run the model
    print("Matrices set up. Attempting to compile...", os.path.join(qthome, "qtbirds.tppl/qtbirds.tppl"), sweep_samples, mthd)
    with treeppl.Model(filename=os.path.join(qthome, "qtbirds.tppl/qtbirds.tppl"), samples=sweep_samples, method=mthd) as qtbirds:
        print("Model compiled. Running inference with", sweep_samples, "samples/particles and", mthd);
        while len(lambda_samples) < total_samples:
            start = timer()
            res = qtbirds(tree=tree, normQChar=norm_q_char, jChar=jChar, charMessages=startMessages,
                          normQMol=norm_q_mol, jMol=jMol,
                          lamShape=prior['lam']['shape'], lamScale=prior['lam']['scale'],
                          muShape=prior['mu']['shape'], muScale=prior['mu']['scale'],
                          nuShape=prior['nu']['shape'], nuScale=prior['nu']['scale'])
            oss = optimal_subsample_size(res)
            end = timer()
            print("Sweep completed. OSS = ", oss)
            print("Seconds per sample: ", (end - start)/oss)

            # Extract samples and log weights
            subsamples = res.subsample(oss)

            #print("Subsamples structure:", subsamples)  # Add this line for debugging

            for sample in subsamples:
                lambda_samples.append(sample[0])  # Extract lambda
                mu_samples.append(sample[1])      # Extract mu
                nu_samples.append(sample[2])      # Extract nu

            lweights.extend([res.norm_const] * oss)

            print(f"So far {len(lambda_samples)} samples; present run log Z = {res.norm_const}; total var log Z = {np.var(lweights)}")

    return lambda_samples, mu_samples, nu_samples, lweights, tree_label


def qtbirds_sim_to_phyjson(tree_data):
    def process_node(node, parent_age=0):
        """
        Recursive function to process each node and construct the PhyJSON tree structure.
        :return: A JSON object encoded as PhyJSON.
        """
        if node["type"] == "leaf":
            # Return leaf node information
            return {
                "taxon": node["index"],
                "branch_length": parent_age - node["age"]
            }
        else:
            # Process left and right children
            left_child = process_node(node["left"], node["age"])
            right_child = process_node(node["right"], node["age"])
            return {
                "children": [left_child, right_child],
                "branch_length": parent_age - node["age"]
            }

    def convert_to_phyjson(input_json):
        phyjson = {
            "format": "phyjson",
            "version": "1.0",
            "taxa": [],
            "characters": [
                {"id": "dna", "type": "dna", "aligned": False},
                {"id": "state", "type": "standard", "symbols": ["black", "white"]}
            ],
            "trees": []
        }

        # Process each tree in the input JSON
        for item in input_json:
            tree_root = process_node(item["value"], item["value"]["age"])
            phyjson["trees"].append({
                "name": "Generated Tree",
                "rooted": True,
                "root": tree_root
            })

            # Extract taxa information from the leaf nodes
            extract_taxa(item["value"], phyjson["taxa"])

        return phyjson

    def extract_taxa(node, taxa):
        """
        Recursive function to extract taxa from the node.
        """
        if node["type"] == "leaf":
            taxa.append({
                "id": node["index"],
                "name": f"Taxon {node['index']}",
                "characters": {
                    "dna": node["sequence"],
                    "state": node["character"]
                }
            })
        elif node["type"] == "node":
            extract_taxa(node["left"], taxa)
            extract_taxa(node["right"], taxa)


    # Your input JSON
    with open(tree_data, 'r') as file:
        input_json = json.load(file)
        
    #input_json_str = '[{"value": {...}}]'  # Replace with your actual JSON string
    #input_json = json.loads(input_json_str)
    phyjson = convert_to_phyjson(input_json)

    # Convert to JSON string for display
    phyjson_str = json.dumps(phyjson, indent=4)
    #print(phyjson_str)
    return phyjson


def phyjson_to_newick(node, taxa_map):
    """
    Recursive function to convert a PhyJSON tree node into Newick format.
    Uses taxa names instead of IDs.
    :return: An array containing Newick strings.
    """
    if 'taxon' in node:
        # Leaf node - use the taxon name from taxa_map
        taxon_name = taxa_map.get(node['taxon'], f"Unknown_{node['taxon']}")
        return f"{taxon_name}:{node['branch_length']}"
    else:
        # Internal node
        children_newick = ','.join([phyjson_to_newick(child, taxa_map) for child in node['children']])
        return f"({children_newick}):{node['branch_length']}"

def convert_trees_to_newick(phyjson_object):
    """
    Convert all trees in the PhyJSON 'trees' array to Newick format.
    Uses names from the taxa list instead of IDs.
    """
    # Create a map of taxon IDs to their names
    taxa_map = {taxon['id']: taxon['name'] for taxon in phyjson_object['taxa']}

    newick_trees = []
    for tree in phyjson_object['trees']:
        newick_tree = phyjson_to_newick(tree['root'], taxa_map) + ';'
        newick_trees.append(newick_tree)
    return newick_trees

    # Example usage:
    # Assuming 'phyjson_object' is your entire PhyJSON object
    # newick_trees = convert_trees_to_newick(phyjson_object)

    # 'newick_trees' now contains Newick representations using taxon names



def taxa_to_dataframe(phyjson_object):
    """
    Convert the taxa information from PhyJSON format to a pandas DataFrame.
    """
    # Extract taxa data
    taxa_data = [{
        "name": taxon['name'],
        "state": taxon['characters']['state'],
        "dna": taxon['characters']['dna']
    } for taxon in phyjson_object['taxa']]

    # Create DataFrame
    df = pd.DataFrame(taxa_data, columns=["name", "state", "dna"])
    return df

    # Example usage:
    # Assuming 'phyjson_object' is your entire PhyJSON object
    # taxa_df = taxa_to_dataframe(phyjson_object)

    # 'taxa_df' is a pandas DataFrame with the taxa information
