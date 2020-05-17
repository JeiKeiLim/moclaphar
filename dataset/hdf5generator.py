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
        label_name= "{}_label".format(self.prefix)

        self.n_class = len(np.unique(f[label_name][()]))
        hist = np.histogram(f[label_name][()], bins=self.n_class - 1)
        self.class_weight = (1 / hist[0]) * f[label_name].shape[0] / self.n_class

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
