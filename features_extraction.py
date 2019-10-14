import numpy as np
import math


def segregate(window):
    rss_list = []
    x_list = []
    y_list = []
    z_list = []
    for i in range(len(window)):
        x_val = window[i][0]
        y_val = window[i][1]
        z_val = window[i][2]
        x_list.append(x_val)
        y_list.append(y_val)
        z_list.append(z_val)
        root_sum_squared = math.sqrt((x_val * x_val + y_val * y_val + z_val * z_val))
        rss_list.append(root_sum_squared)

    return x_list, y_list, z_list, rss_list


def minimum(window):
    min_list = []
    x, y, z, rss_list = segregate(window)
    min_list.append(np.min(x))
    min_list.append(np.min(y))
    min_list.append(np.min(z))
    min_list.append(np.min(rss_list))
    return min_list


def maximum(window):
    max_list = []
    x, y, z, rss_list = segregate(window)
    max_list.append(np.max(x))
    max_list.append(np.max(y))
    max_list.append(np.max(z))
    max_list.append(np.max(rss_list))

    return max_list


def stddev(window):
    std_list = []
    x, y, z, rss_list = segregate(window)
    std_list.append(np.std(x))
    std_list.append(np.std(y))
    std_list.append(np.std(z))
    std_list.append(np.std(rss_list))

    return std_list


def mean(window):
    mean_list = []
    x, y, z, rss_list = segregate(window)
    mean_list.append(np.mean(x))
    mean_list.append(np.mean(y))
    mean_list.append(np.mean(z))
    mean_list.append(np.mean(rss_list))

    return mean_list


# Order of features: min x y z rss | max x y z rss | std x y z rss | mean x y z rss
def extract_features(window):
    return minimum(window) + maximum(window) + stddev(window) + mean(window)


