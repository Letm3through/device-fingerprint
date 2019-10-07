import numpy as np
import math


def rss(window):
    rss_list = []
    for i in range(len(window)):
        x_val = window[i][0]
        y_val = window[i][1]
        z_val = window[i][2]
        root_sum_squared = math.sqrt((x_val * x_val + y_val * y_val + z_val * z_val))
        rss_list.append(root_sum_squared)

    return rss_list


def minimum(window):
    rss_list = rss(window)
    minimum_val = np.min(rss_list)
    return minimum_val


def maximum(window):
    rss_list = rss(window)
    maximum_val = np.min(rss_list)
    return maximum_val


def stddev(window):
    rss_list = rss(window)
    std_dev = np.std(rss_list)
    return std_dev


def extract_features(window):
    features = [minimum(window), maximum(window), stddev(window)]
    return features
