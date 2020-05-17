import h5py
import numpy as np
import tensorflow as tf


class HDF5Generator:
    def __init__(self, path, prefix, verbose=1):
        self.path = path
        self.prefix = prefix
        self.data = h5py.File(self.path, 'r')
        f = self.data

        self.n_data = f["{}_data".format(self.prefix)].shape[0]
        label_name = "{}_label".format(self.prefix)

        labels = f[label_name][()]
        unique_labels = np.unique(labels)

        self.class_histogram = np.array([labels[labels == i].shape[0] for i in unique_labels])
        self.n_class = unique_labels.shape[0]
        c_weight = (1 / self.class_histogram) * labels.shape[0] / self.n_class
        self.class_weight = {i: weight for i, weight in enumerate(c_weight)}

        if verbose > 0:
            print("=+*"*20)
            print("Dataset {} with {} data".format(path, self.n_data))
            print("=+*"*20)

    def __call__(self):
        with h5py.File(self.path, 'r') as f:
            for d, l in zip(f["{}_data".format(self.prefix)], f["{}_label".format(self.prefix)]):
                yield (d, l)

    def get_dataset(self, input_shape=(300, 30), batch_size=32, shuffle=False, n_shuffle=10000):
        d_set = tf.data.Dataset.from_generator(
            self,
            (tf.float32, tf.int8),
            (tf.TensorShape(input_shape), tf.TensorShape([]))
        )
        d_set = d_set.shuffle(n_shuffle, reshuffle_each_iteration=True) if shuffle else d_set

        return d_set.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).repeat()

    def nan_check(self):
        key = "{}_data".format(self.prefix)

        is_nan = False
        for i in range(self.data[key].shape[0]):
            if np.sum(np.isnan(self.data[key][i].flatten())) > 0:
                print("NaN Exist! on {}".format(i))
                is_nan = True
        return is_nan
