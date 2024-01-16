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

def optimal_subsample_size(inference_result):
    def compress_samples(samples, nweights):
        unique_samples = {}

        for sample, weight in zip(samples, nweights):
            sample_tuple = tuple(sample)

            if sample_tuple in unique_samples:
                unique_samples[sample_tuple] += weight
            else:
                unique_samples[sample_tuple] = weight

        compressed_samples = [list(sample) for sample in unique_samples.keys()]
        compressed_nweights = list(unique_samples.values())

        return compressed_samples, compressed_nweights

    def calculate_ess(nweights):
        if nweights is None or len(nweights) == 0:
            return 0

        normalized_weights = nweights / np.sum(nweights)
        sum_of_squares = np.sum(normalized_weights**2)
        return 1 / sum_of_squares

    compressed_samples, compressed_nweights = compress_samples(inference_result.samples, inference_result.nweights)
    ess = calculate_ess(compressed_nweights)
    return Math.ceil(ess)


def find_MAP(values, weights):
    # Normalize the weights to avoid numerical instability
    normalized_weights = np.exp(weights - np.max(weights))

    # Create a KDE using the values and normalized weights
    kde = gaussian_kde(values, weights=normalized_weights)

    # Function to return the negative of KDE (since we want to maximize the KDE)
    def neg_kde(x):
        return -kde(x)[0]

    # Find the maximum of the KDE
    result = minimize_scalar(neg_kde, bounds=(min(values), max(values)), method='bounded')

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization did not converge")
    

def find_mean(values, weights):
    # Normalize the weights to sum up to 1 (to handle any scale of weights)
    normalized_weights = weights / np.sum(weights)

    # Calculate the weighted mean
    weighted_mean = np.sum(values * normalized_weights)

    return weighted_mean




def compute_hdpi(samples, log_weights, hdpi_prob=0.95):
    # Convert log weights to linear scale
    linear_weights = np.exp(log_weights - np.max(log_weights))

    # Perform weighted KDE
    kde = gaussian_kde(samples, weights=linear_weights)

    # Compute the HDPI
    # This step involves finding the densest interval containing hdpi_prob of the probability mass
    # It requires custom implementation, as scipy's gaussian_kde doesn't provide a direct method for this
    
    # Determine the range for the grid points
    sample_std = np.std(samples, ddof=1)  # Sample standard deviation
    bandwidth = kde.factor * sample_std  # Approximate bandwidth used by KDE
    grid_min = min(samples) - 3 * bandwidth
    grid_max = max(samples) + 3 * bandwidth

    # Compute the HDPI
    grid_points = np.linspace(grid_min, grid_max, 1000)

    kde_values = kde(grid_points)
    sorted_indices = np.argsort(kde_values)[::-1]  # Sort by density

    cumulative_prob = 0
    interval_indices = []
    for idx in sorted_indices:
        interval_indices.append(idx)
        cumulative_prob += kde_values[idx] / sum(kde_values)
        if cumulative_prob >= hdpi_prob:
            break

    interval_indices = np.sort(interval_indices)
    hdpi_interval = (grid_points[interval_indices[0]], grid_points[interval_indices[-1]])

    return hdpi_interval


def find_min_hdpi_prob(x, samples, log_weights):
    hdpi_prob = 0.01  # Start with the smallest interval probability
    max_hdpi_prob = 1.00  # Maximum interval probability
    resolution = 0.01  # Increment resolution

    while hdpi_prob <= max_hdpi_prob:
        current_hdpi = compute_hdpi(samples, log_weights, hdpi_prob=hdpi_prob)

        if current_hdpi[0] <= x <= current_hdpi[1]:
            # x is within the interval, return the current interval and its probability
            return hdpi_prob, current_hdpi[0], current_hdpi[1]

        # Increase the interval probability
        hdpi_prob += resolution

    # If the loop completes without returning, x is not in any interval.
    # This is unlikely but could happen if x is an outlier or if there's an issue with the data.
    return None, current_hdpi[0], current_hdpi[1]


def find_min_hdpi_prob_bin(x, samples, log_weights):
    low = 0.01  # Lower bound of the search range
    high = 1.00  # Upper bound of the search range
    resolution = 0.01  # Desired resolution for the search

    while high - low > resolution:
        mid = (high + low) / 2
        current_hdpi = compute_hdpi(samples, log_weights, hdpi_prob=mid)

        if current_hdpi[0] <= x <= current_hdpi[1]:
            # x is within the interval, try a smaller interval
            high = mid
        else:
            # x is not within the interval, try a larger interval
            low = mid

    # Compute the final HDPI at the lower bound of the last search interval
    # This ensures that the HDPI is the tightest one containing x
    final_hdpi = compute_hdpi(samples, log_weights, hdpi_prob=high)
    return high, final_hdpi[0], final_hdpi[1]


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
