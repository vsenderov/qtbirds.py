import unittest
import numpy as np
from .utils import compute_hdpi, find_min_hdpi_prob, find_min_hdpi_prob_bin  # Adjust the import path as necessary

class TestComputeHDI(unittest.TestCase):

    def test_hdi_normal_distribution(self):
        # Test with a standard normal distribution
        samples = np.random.normal(0, 1, 1000)
        log_weights = np.zeros_like(samples)  # Equal weights
        expected_hdi_range = 2 * 1.96  # For a normal distribution, approximately 95% data lies within +/- 1.96 std deviations

        hdi_low, hdi_high = compute_hdpi(samples, log_weights, hdpi_prob=0.95)
        hdi_width = hdi_high - hdi_low

        # Check if the computed HDI width is approximately equal to the expected range
        self.assertAlmostEqual(hdi_width, expected_hdi_range, delta=0.5)

    def test_hdi_with_weights(self):
        # Test with weighted samples
        samples = np.random.normal(0, 1, 1000)
        log_weights = np.random.uniform(-1, 1, 1000)  # Random weights

        # Since we don't have a specific expected value, we check if the function executes without errors
        hdi_interval = compute_hdpi(samples, log_weights, hdpi_prob=0.95)
        self.assertTrue(len(hdi_interval) == 2)
        self.assertTrue(hdi_interval[0] <= hdi_interval[1])

    # def test_hdi_edge_case(self):
    #     # Test with edge case (e.g., all samples are identical)
    #     samples = np.ones(1000)
    #     log_weights = np.zeros_like(samples)  # Equal weights

    #     # The HDI should be a single point in this case
    #     hdi_interval = compute_hdpi(samples, log_weights, hdpi_prob=0.95)
    #     self.assertEqual(hdi_interval[0], hdi_interval[1])

    # You can add more tests to cover other cases or scenarios

    def test_find_min_hdpi_prob_normal(self):
        # Test with a standard normal distribution and a known value of x within that distribution
        samples = np.random.normal(0, 1, 1000)
        log_weights = np.zeros_like(samples)  # Equal weights
        x = 0  # Known value within the distribution

        # Find the minimal HDPI probability that contains x
        hdpi_prob, hdpi_low, hdpi_high = find_min_hdpi_prob(x, samples, log_weights)
        hdpi_prob_bin, hdpi_low_bin, hdpi_high_bin = find_min_hdpi_prob_bin(x, samples, log_weights)

        # Check if the returned HDPI probability is valid and the interval includes x
        self.assertIsNotNone(hdpi_prob)
        #print(hdpi_prob, hdpi_low, hdpi_high)
        #print(hdpi_prob_bin, hdpi_low_bin, hdpi_high_bin)
        self.assertEqual(int(hdpi_prob), int(hdpi_prob_bin))
        self.assertGreaterEqual(x, hdpi_low)
        self.assertLessEqual(x, hdpi_high)

       # Expecting a very tight HDPI since x is the mean of the distribution
        self.assertLess(hdpi_prob, 0.34)  # Expecting HDPI to be much tighter than 68%

    # Add more test cases to cover different scenarios, edge cases, and special conditions

if __name__ == '__main__':
    unittest.main()
