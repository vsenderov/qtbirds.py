import json
import numpy as np
import math as Math
import arviz as az
from .QTTree import QTNode, QTLeaf
from scipy.stats import gaussian_kde
from scipy.optimize import minimize_scalar

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



def compute_hdpi(samples, log_weights, hdpi_prob=0.95):
    # Convert log weights to linear scale and normalize
    linear_weights = np.exp(log_weights - np.max(log_weights))
    normalized_weights = linear_weights / np.sum(linear_weights)

    # Resample the samples based on the weights
    weighted_samples = np.random.choice(samples, size=len(samples), p=normalized_weights)

    # Compute the HDPI using arviz
    hdpi_interval = az.hdi(weighted_samples, hdi_prob=hdpi_prob)

    return hdpi_interval


def find_min_hdpi_prob(x, samples, log_weights):
    low = 0.01
    high = 1.00
    resolution = 0.01
    min_hdpi_prob = None

    while high - low > resolution:
        mid = (high + low) / 2
        current_hdpi = compute_hdpi(samples, log_weights, hdpi_prob=mid)

        if current_hdpi[0] <= x <= current_hdpi[1]:
            # x is within the interval, try narrowing it
            high = mid
            min_hdpi_prob = mid
        else:
            # x is not within the interval, need to widen it
            low = mid

    return min_hdpi_prob


