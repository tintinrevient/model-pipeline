from scipy.stats import entropy
import matplotlib.pyplot as plt
import numpy as np


def data_length_normalizer(gt_data, obs_data, bins=100):
    """
    Data length normalizer will normalize a set of data points if they
    are not the same length.

    params:
        gt_data (List) : The list of values associated with the training data
        obs_data (List) : The list of values associated with the observations
        bins (Int) : The number of bins you want to use for the distributions

    returns:
        The ground truth and observation data in the same length.
    """

    if len(gt_data) == len(obs_data):
        return gt_data, obs_data

        # scale bins accordingly to data size
    if (len(gt_data) > 20 * bins) and (len(obs_data) > 20 * bins):
        bins = 10 * bins

        # convert into frequency based distributions
    gt_hist = plt.hist(gt_data, bins=bins)[0]
    obs_hist = plt.hist(obs_data, bins=bins)[0]
    plt.close()  # prevents plot from showing
    return gt_hist, obs_hist


def softmax(vec):
    """
    This function will calculate the softmax of an array, essentially it will
    convert an array of values into an array of probabilities.

    params:
        vec (List) : A list of values you want to calculate the softmax for

    returns:
        A list of probabilities associated with the input vector
    """
    return (np.exp(vec) / np.exp(vec).sum())


def calc_cross_entropy(p, q):
    """
    This function will calculate the cross entropy for a pair of
    distributions.

    params:
        p (List) : A discrete distribution of values
        q (List) : Sequence against which the relative entropy is computed.

    returns:
        The calculated entropy
    """
    return entropy(p, q)


def calc_drift(gt_data, obs_data, gt_col, obs_col):
    """
    This function will calculate the drift of two distributions given
    the drift type identifeid by the user.

    params:
        gt_data (DataFrame) : The dataset which holds the training information
        obs_data (DataFrame) : The dataset which holds the observed information
        gt_col (String) : The training data column you want to compare
        obs_col (String) : The observation column you want to compare

    returns:
        A drift score
    """

    gt_data = gt_data[gt_col].values
    obs_data = obs_data[obs_col].values

    # makes sure the data is same size
    gt_data, obs_data = data_length_normalizer(
        gt_data=gt_data,
        obs_data=obs_data
    )

    # convert to probabilities
    gt_data = softmax(gt_data)
    obs_data = softmax(obs_data)

    # run drift scores
    drift_score = calc_cross_entropy(gt_data, obs_data)
    return drift_score