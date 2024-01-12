from math import sqrt

import numpy as np

from anomalie_detection.BrutForce.brut_force import compute_score


def sort_frames_brutForce(clean_data,main_cluster_index):
    data = clean_data
    for d in data[:]:
        data.append(generate_data_for_transpose(d))

    # Attempt 2
    img_orders = []
    median_weights = []
    for i in main_cluster_index:

        frames = list(filter(lambda x: x['frames'][0] == i, data)) + [{
            "frames": [i, i],
            "mean": 0,
            "median": 0,
            "matches": 0,
            "fraction": 0,
            "x": 0,
            "y": 0,
        }]
        sorted_frames = sorted(frames, key=lambda x: np.sign(x['x']) * distance(x))
        img_order = [x['frames'][1] for x in sorted_frames]
        median_weight = [1 / compute_score(x) if compute_score(x) != 0 else 0 for x in sorted_frames]
        img_orders.append(img_order)
        median_weights.append(median_weight)

    np_orders = np.array(img_orders)
    np_weights = np.array(median_weights)
    img_order = []
    for i in range(len(np_orders)):
        img_order += [most_common_weighted(list(np_orders[:, i]), list(np_weights[:, i]))]
    return img_order

def generate_data_for_transpose(d):
    """ Take a frame (i, j) and returns an entry for frame (j, i). """
    new_d = dict(d)
    new_d['frames'] = d['frames'][::-1]
    new_d['x'] = - d['x']
    new_d['y'] = - d['y']
    return new_d

def distance(data):
    """ Compute euclidian distance. """
    return sqrt(data['x'] ** 2 + data['y'] ** 2)


def most_common(arr):
    """ Return the most frequent occurence in an array. """
    return max(set(arr), key=arr.count)
def most_common_weighted(arr, weights):
    """ Returns the most frequent occurence in array. The frequency
    is weighted by the array of weights. """
    frequencies = {}
    for i, e in enumerate(arr):
        if e not in frequencies.keys():
            frequencies[e] = weights[i]
        else:
            frequencies[e] += weights[i]
    return max(frequencies, key=frequencies.get)