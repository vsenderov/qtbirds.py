import subprocess
import json
import uuid
import os

def run_simulation(qt_home, dep_home, fasta_file="phylo/sample2.fasta", 
                   tree_file="phylo/jeremy-crbd.tre.phyjson", 
                   states="pheno/2state.json", nucleo_json="nucleo/jc69.Q.json", 
                   pheno_json="pheno/mk-2state.Q.json", lam=0.1, mu=0.1, nu=0.0):
    """
    This function does a Q-T-Birds simulation by invoking the WebPPL simulator
    via a subprocess command.

    Side-effects: two files, one with rates, and another with the simulation.

    :param qt_home: The directory where the simulator is installed
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
    command = f"webppl {os.path.join(qt_home, 'qtbirds-sim.wppl')} --require {os.path.join(dep_home, 'fasta2json')} " \
              f"--require {qt_home} --require webppl-fs --require {os.path.join(dep_home, 'phywppl/phyjs')} -- " \
              f"{os.path.join(qt_home, fasta_file)} {os.path.join(qt_home, tree_file)} {rates_filename} " \
              f"{os.path.join(qt_home, states)} {os.path.join(qt_home, nucleo_json)} " \
              f"{os.path.join(qt_home, pheno_json)} 1 > {output_filename}"

    print(command)
    # Execute the command
    subprocess.run(command, shell=True)

    # Clean up the rates file
    # os.remove(rates_filename)  # Uncomment this line if you want to delete the rates file after execution

    print(f"Simulation output saved in {output_file}")
    return output_filename

# Example usage
# output_file = run_simulation("/home/viktor/Sync/Workspaces/Q-T-Birds/dep/qtbirds.webppl.sp13", "/home/viktor/Sync/Workspaces/Q-T-Birds/dep")

