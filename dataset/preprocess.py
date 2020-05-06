import numpy as np


def normalize_data(data, norm_max=1):
    min_v = np.ones(data[0].shape[1:]) * np.inf
    max_v = np.ones(data[0].shape[1:]) * -np.inf

    for d in data:
        for i in range(d.shape[1]):
            for j in range(d.shape[2]):
                min_v[i, j] = min(min_v[i, j], d[:, i, j].min())
                max_v[i, j] = max(max_v[i, j], d[:, i, j].max())

    for d in data:
        for i in range(d.shape[1]):
            for j in range(d.shape[2]):
                d[:, i, j] = ((d[:, i, j] - min_v[i, j]) / (max_v[i, j]-min_v[i, j])) * norm_max

    return data


def reshape_data(data, rotate=True):
    data = data.reshape(data.shape[0], data.shape[1], data.shape[2]*data.shape[3])
    if rotate:
        data = np.rot90(data, axes=(1,2))

    return data


def generate_sliding_window_data(data, label, window_size=300, stride=1):
    n_data = 0
    for d, l in zip(data, label):
        if d.shape[0] < window_size:
            continue

        for i in range(0, d.shape[0] - window_size, stride):
            n_data += 1

    w_data = np.zeros((n_data, window_size, data[0].shape[1], data[0].shape[2]))
    w_label = np.zeros((n_data, ))
    idx = 0
    for d, l in zip(data, label):
        if d.shape[0] < window_size:
            continue

        for i in range(0, d.shape[0]-window_size, stride):
            w_data[idx] = d[i:(i+window_size)]
            w_label[idx] = l
            idx += 1

    return w_data, w_label
