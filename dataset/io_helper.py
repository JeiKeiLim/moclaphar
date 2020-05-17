import h5py
import numpy as np
from .preprocess import generate_sliding_window_data
from .preprocess import reshape_data


def append_h5py_data(data, fname, db_key, dtype='float32'):
    with h5py.File(fname, "a") as f:
        if db_key in f.keys():
            f[db_key].resize(f[db_key].shape[0] + data.shape[0], axis=0)
            f[db_key][-data.shape[0]:] = data
        else:
            maxshape = list(data.shape)
            maxshape[0] = None
            f.create_dataset(db_key, data=data, dtype=dtype, maxshape=tuple(maxshape), chunks=True)


def save_windowed_data_hdf5(data, label, s_idx=-1, e_idx=-1, window_size=300, stride=1, save_root='../data/', prefix="training", verbose=1):
    if s_idx < 0 or e_idx < 0:
        return

    t_data, t_label = data[s_idx:e_idx], label[s_idx:e_idx]

    if verbose > 0:
        print("Generating sliding window ... {} ~ {}".format(s_idx, e_idx))

    t_data, t_label = generate_sliding_window_data(t_data, t_label, window_size=window_size, stride=stride)
    t_data = reshape_data(t_data, rotate=False)
    t_label = t_label.astype("int8")
    t_data = t_data.astype('float32')

    fname_prefix = "{}{}".format(save_root, prefix)

    append_h5py_data(t_data, "{}_data.hdf5".format(fname_prefix), "{}_data".format(prefix), dtype='float32')
    append_h5py_data(t_label, "{}_data.hdf5".format(fname_prefix), "{}_label".format(prefix), dtype='int8')


def save_windowed_dataset_hdf5(training_data, training_label, test_data, test_label,
                               window_size=300, stride=1, save_root='../data/',
                               chunk_size=100):
    for i in range(0, len(training_data), chunk_size):
        s_idx = i
        e_idx = min(i + chunk_size, len(training_data))
        save_windowed_data_hdf5(training_data, training_label, s_idx, e_idx, window_size=window_size,
                                stride=stride, save_root="{}".format(save_root), prefix="training")

    for i in range(0, len(test_data), chunk_size):
        s_idx = i
        e_idx = min(i + chunk_size, len(test_data))
        save_windowed_data_hdf5(test_data, test_label, s_idx, e_idx, window_size=window_size,
                                stride=stride, save_root="{}".format(save_root), prefix="test")

if __name__ == "__main__":
    pass
