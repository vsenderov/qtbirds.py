import json
import numpy as np
import math as Math
from .QTTree import QTNode, QTLeaf
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar
from scipy.stats import gaussian_kde
import numpy as np
import os

type_map = {
    "node": QTNode,
    "leaf": QTLeaf
}

def object_hook(value):
    if value.get("type") in type_map:
        type_key = value.pop("type")
        return type_map[type_key](**value)
    return value

def load_data(filename):
    with open(filename) as f:
        return json.load(f, object_hook=object_hook)

def save_newick_string(newick, output_file):
    """
    Save the first Newick tree string to disk with a new filename.
    """
    # Extract the first Newick string (assuming there's only one tree)
    newick_string = newick[0]

    # Create a new filename with a Newick format extension
    base_name = os.path.splitext(output_file)[0]
    newick_filename = base_name + '.nwk'

    # Save the Newick string to the new file
    with open(newick_filename, 'w') as file:
        file.write(newick_string)
    
    return newick_filename

# Example usage:
# newick = convert_trees_to_newick(tree_phyjson)
# output_file = 'output-817bb7d2-10f0-4282-bbd3-bf447304974f.json'
# save_newick_string(newick, output_file)

# This will save the Newick string in a file with a filename based on 'output_file'


def save_dataframe_to_csv(dataframe, output_file):
    """
    Save the DataFrame to a CSV file, using a filename based on 'output_file'.
    """
    # Create a new filename with a .csv extension
    base_name = os.path.splitext(output_file)[0]
    csv_filename = base_name + '.csv'

    # Save the DataFrame to the new file
    dataframe.to_csv(csv_filename, index=False)
    return csv_filename

# Example usage:
# dataframe = taxa_to_dataframe(tree_phyjson)
# output_file = 'output-817bb7d2-10f0-4282-bbd3-bf447304974f.json'
# save_dataframe_to_csv(dataframe, output_file)

# This will save the DataFrame in a CSV file with a filename based on 'output_file'


# def compute_probabilities(rho_samples, lweights):
#     print("got: ", rho_samples, lweights)
#     # Ensure rho_samples is a numpy array with float64 data type
#     rho_samples = np.array(rho_samples, dtype=np.float64)
    
#     # Use the log-sum-exp trick for numerical stability
#     max_lweight = np.max(lweights)
#     weights = np.exp(lweights - max_lweight)
#     print("weights after exp: ", weights)

#     # Compute the total weight for normalization
#     total_weight = np.sum(weights)
#     print("total weight: ", total_weight)

#     # Check for division by zero
#     if total_weight == 0:
#         raise ValueError("Total weight is zero, check lweights for large negative values")

#     # Normalize the weights to avoid dividing by zero
#     weights /= total_weight
#     print("normalized weights: ", weights)

#     # Check what the comparisons are returning
#     print("rho_samples == 0.0: ", rho_samples == 0.0)
#     print("rho_samples == 1.0: ", rho_samples == 1.0)

#     # Compute weighted probabilities for False (0) and True (1)
#     prob_false = np.sum(weights[rho_samples == 0.0])
#     prob_true = np.sum(weights[rho_samples == 1.0])
    
#     print("returning:", prob_false, prob_true)

#     return prob_false, prob_true


def check_os_environment():
    import os

    # Check if QTHOME is set
    if 'QTHOME' not in os.environ:
        print("QTHOME environment variable is not set. Please set it and try again.")
        exit()

    # Check if MCORE_LIBS is set
    if 'MCORE_LIBS' not in os.environ:
        print("MCORE_LIBS environment variable is not set. Please set it and try again.")
        exit()
        
    # Check if QT_WPPL_HOME is set
    if 'QT_WPPL_HOME' not in os.environ:
        print("QT_WPPL_HOME environment variable is not set. Please set it and try again.")
        exit()
        
    # Check if QT_DEP_HOME is set
    if 'QT_DEP_HOME' not in os.environ:
        print("QT_DEP_HOME environment variable is not set. Please set it and try again.")
        exit()

    # Split MCORE_LIBS by ':' to get individual clauses
    clauses = os.environ['MCORE_LIBS'].split(':')

    # Extract the key part before '=' from each clause
    clause_keys = [clause.split('=')[0] for clause in clauses]

    # Check if clauses for coreppl, stdlib, and treeppl are present
    required_clauses = ['coreppl', 'stdlib', 'treeppl']
    missing_clauses = [clause for clause in required_clauses if clause not in clause_keys]

    if missing_clauses:
        print(f"The following clauses are missing in MCORE_LIBS: {', '.join(missing_clauses)}")
        exit()

    # If all checks pass, you can proceed with your code
    print("All environment variable checks passed. You can proceed with your code.")
    return [os.environ['QTHOME'], os.environ['MCORE_LIBS'], os.environ['QT_WPPL_HOME'], os.environ['QT_DEP_HOME']]



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





def read_external_data(nexus_file, fasta_file, csv_file):
    """
    Merges the first Newick Tree from a NEXUS file with sequence data from a FASTA file,
    and trait information from a CSV file. It matches sequences and traits to tree leaves 
    based on species names.
    
    Parameters:
    - nexus_file: Path to the NEXUS file containing phylogenetic trees.
    - fasta_file: Path to the FASTA file containing sequence data for various species.
    - csv_file: Path to the CSV file containing species names and their traits.
    
    Returns:
    - A QTNode object representing the root of the phylogenetic tree with sequences and 
      traits integrated.
    
    The function reads species traits from the CSV file, matches them with sequence data 
    from the FASTA file by species name, and integrates this information into the 
    phylogenetic tree structure obtained from the NEXUS file. The result is a fully 
    constructed QTNode object that represents the root of the tree with all data 
    integrated.
    """
    from Bio import SeqIO, Phylo
    import pandas as pd

    # Read species traits from the CSV file
    characters = pd.read_csv(csv_file, sep='\t', header=0, index_col=None, na_values=['NA', '?'])
    
    def get_index(name):
        """Retrieve the index of the species in the CSV file."""
        index = characters.index[characters['species'] == name].tolist()
        if index:
            return index[0]
        print("ERROR: Species not found!")
        return None

    def get_character(name):
        """Retrieve the character trait of the species from the CSV file."""
        ch = characters['trait'][characters['species'] == name].tolist()
        if ch:
            return ch[0]
        print("ERROR: Species not found!")
        return None

    # Read the FASTA file for sequence data and match by name
    sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")}
    
    def calculate_node_age(clade):
        if not clade.clades:
            return 0.0
        return clade.branch_length + max(calculate_node_age(child) for child in clade.clades)
    
    def convert_clade(clade):
        """
        Recursive function to convert a Bio.Phylo clade into QTNode or QTLeaf objects,
        incorporating sequence and trait information based on species name.
        """
        # Base case: clade is a leaf
        node_age = calculate_node_age(clade)
        if not clade.clades:
            return QTLeaf(age=0.0, index=get_index(clade.name),
                          sequence=sequences[clade.name], characterState=get_character(clade.name),
                          character=get_character(clade.name))
        # Recursive case: clade is an internal node
        else:
            children = [convert_clade(child) for child in clade.clades]
            return QTNode(left=children[0], right=children[1], age=node_age)

    def convert_phylo_tree_to_qt_tree(phylo_tree):
        """Converts the root clade of a Bio.Phylo tree to a QTNode tree."""
        root_clade = phylo_tree.root
        return convert_clade(root_clade)

    # Read the first phylogenetic tree from the NEXUS file
    phylo_tree = next(Phylo.parse(nexus_file, "nexus"))
    # Convert the phylogenetic tree to a QTNode tree
    qt_tree = convert_phylo_tree_to_qt_tree(phylo_tree)

    return qt_tree
