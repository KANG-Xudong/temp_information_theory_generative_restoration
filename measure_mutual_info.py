import warnings
import numpy as np
from scipy.stats import entropy


class InfoMeasures():
    def __init__(self, is_uniform=False, base='e'):
        # specify whether the values of both input vectors are uniformly distributed
        # specify whether all possible events have equal probabilities (uniformly distributed)
        
        if base == 'e':
            self.log = lambda x: np.log(x + (prob == 0)) # logarithm is set to zero for zero probability
        else:
            self.log = lambda x : np.log(x + (prob == 0)) / np.log(base) # logarithm base change rule

    def marginal_entropy(self, xs, is_uniform=False, num_values=None):
        assert np.array(xs).ndim == 2, "Invalid input shape: xs.shape={}.".format(xs.shape)
        num_samples, vector_length = xs.shape
        
        if is_uniform:
        	# if the random vector has a uniform distribution, any instance vectors in the input xs has equal probability,
        	# which equals to (1 / num_values)^vector_length, and thus marginal entropy of the random vector can be 
        	# measured by calculating the self-information of any of its instance vector
        	unique_values = np.unique(xs)
            if num_values is not None:
                assert isinstance(num_values, int), "Invalid input: num_values={}.".format(num_values)
                assert num_values >= len(unique_values), \
                    "Actual number of unique values larger than the specified number: {}".format(unique_values)
                return vector_length * self.log(num_values) # self-information I = marginal_entropy H = L * log K
            else:
            	# if the number of possible values are not given, we can estimate it by using the unique values that appears in the instances inputs
            	# however, since not all possible values of the random vector may appears in its instances samples,
            	# if the number of samples is too small, the estimated number of possible values may not be so accurate.
                K = len(unique_values) + 1
                assert num_samples > (1 / vector_length) * (np.log(0.99 / K) / np.log(1 - (1 / K))), \
                    "Number of samples is too small, which may under-estimate the number of possible values."
                return vector_length * self.log(len(unique_values))
        else:
        	# if the random vector has a unknown distribution, we can only estimate the possible cases and their probability
        	# based on the limited instance samples provided, where all unique vectors will be grouped and counts,
        	# and the samples' expected self-information will be calculated in estimating the marginal entropy of the random vector
        	unique_vectors, cases_counts = np.unique(xs, return_counts=True, axis=0)
        	if num_values is not None:
                assert isinstance(num_values, int), "Invalid input: num_values={}.".format(num_values)
                assert num_values >= len(np.unique(xs)), \
                    "Actual number of unique values larger than the specified number: {}".format(np.unique(xs))
        		assert len(unique_vectors) <= num_values ** vector_length, \
        	        "Number of unique vectors in the samples exceeds the theoretical number of cases."
        	assert np.sum(cases_counts) == len(xs)
        	probabilities = cases_counts / np.sum(cases_counts)
        	return -np.sum(probabilities * self.log(probabilities))

    def joint_entropy(self, xs, ys):
    	assert xs.shape[0] == ys.shape[0], "Input xs and ys have different number of samples."
    	_, x_indices = np.unique(xs, return_inverse=True, axis=0)
    	_, y_indices = np.unique(ys, return_inverse=True, axis=0)
    	assert len(x_indices) == len(y_indices), "Unexpected error."
    	indices_pairs = np.stack((x_indices, y_indices), axis=1)
    	_, pairs_cases_counts = np.unique(indices_pairs, return_counts=True, axis=0)
    	probabilities = pairs_cases_counts / len(pairs_cases_counts)
    	return -np.sum(probabilities * self.log(probabilities))

    def mutual_information(self, xs, ys):
    	H_X = self.marginal_entropy(xs)
    	H_Y = self.marginal_entropy(ys)
    	H_XY = self.joint_entropy(xs, ys)
    	assert max(H_X, H_Y) <= H_XY and H_XY <= H_X + H_Y, "Unknown error."
    	return H_X + H_Y - H_XY

    def normalized_mutual_information(self, xs, ys):
    	H_X = self.marginal_entropy(xs)
    	H_Y = self.marginal_entropy(ys)
    	H_XY = self.joint_entropy(xs, ys)
    	assert max(H_X, H_Y) <= H_XY and H_XY <= H_X + H_Y, "Unknown error."
    	return H_X + H_Y - H_XY

    def self_info():
    	return





warnings.warn("The number of samples is too small, where the expected values of the measured self-information may under-estimate the information entropy of the random vector.")
                which may under-estimate the amount of information in the input vector.")



    def marginal_entropy(xs, num_values=None):

        value_counts = np.unique(x, return_counts=True)[1]

        if (num_values is not None) and self.are_uniform:
            assert len(value_counts) <= num_values, \
                "The number of unique values exceed the specified number of possible values!"
            return len(x) * self.log(num_values)
        else:
            warnings.warn("The number of possible values are not specified, \
                or the elements in the random vector are non-uniformly distributed. \
                The entropy of the vector is thus estimated based on its histogram, \
                which may under-estimate the amount of information in the input vector.")
            prob = value_counts / len(x)
            try:
                return entropy(prob, base=self.base)
            except:
                warnings.warn("Unexpected error using scipy.stats.entropy.")
                return -np.sum(prob * self.log(prob))

    
    def _joint_entropy(x, y, num_values_x=None, num_values_y=None):
        if None not in (num_values_x, num_values_y):
            hist = np.histogram2d(x, y, bins=[num_values_x + 1, num_values_y + 1])

    def __call__(self, x, y):

