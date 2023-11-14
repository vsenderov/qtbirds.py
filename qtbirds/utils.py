import json
import numpy as np
import math as Math
from .QTTree import QTNode, QTLeaf

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
