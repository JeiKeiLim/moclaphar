try:
    from ..utils.data_loader import read_mat_file
except:
    from utils.data_loader import read_mat_file

from .preprocess import normalize_data
from .io_helper import save_windowed_dataset_hdf5
import numpy as np
import os
import re
import json
from tqdm import tqdm


def get_file_list(root, ext=""):
    f_list = []

    for r, d, f in os.walk(root):
        for fname in f:
            if fname.endswith(ext):
                f_list.append("{}/{}".format(r, fname))
    return f_list


def prepare_data(root, accelerometer=True, gyroscope=True, orientation=False, stroke=False, merge_clap_null=True, verbose=1):
    data = []
    labels = []
    subjects = []

    label_info = dict()
    subject_list = set()
    file_list = get_file_list(root, ext=".mat")

    pbar = tqdm(file_list, desc="Reading mat files ...")
    for i, fpath in enumerate(pbar):
        subject_name = re.search("(S[0-9]+_|[sS]troke[0-9]+)", fpath).group().replace("_", "").lower()

        if (stroke and not subject_name.startswith("stroke")) or \
                (not stroke and subject_name.startswith("stroke")):
            continue

        pbar.set_description("Reading {} ...".format(fpath))

        subject_list.add(subject_name)

        try:
            csv, video, segment = read_mat_file(fpath)

            for sensor_data, label, name in zip(segment['segment_sensor_data'], segment['segment_label'], segment['segment_name']):
                name = name.replace(" ", "").lower().replace("pck", "pick")

                # Fix wrong label START
                erratas = {
                        'clapnull': 'clap',
                        'openrefrigv': 'openrefrig',
                        '청소기밀기': 'vacuum',
                        'clapl': 'clap'
                    }

                if name in erratas.keys():
                    name = erratas[name]
                # Fix wrong label END

                if merge_clap_null:
                    name = name.replace("clap", "null")

                if name not in label_info.keys():
                    label_info[name] = len(label_info)

                data.append(sensor_data)
                subjects.append(subject_name)
                labels.append(label_info[name])

        except ValueError:
            print("Read Error! {}".format(fpath))

    subject_list = sorted(subject_list)

    if verbose > 0:
        print("Label information")
        inv_label = {v: k for k, v in label_info.items()}
        label_hist = np.histogram(labels, bins=len(label_info)-1)

        for hist, label in zip(label_hist[0], label_hist[1]):
            print("{}: {}, ".format(inv_label[label], hist), end="")
        print("")

    sensor_idx = []
    if accelerometer:
        sensor_idx += [0, 1, 2]
    if gyroscope:
        sensor_idx += [3, 4, 5]
    if orientation:
        sensor_idx += [6, 7, 8]

    for i in range(len(data)):
        data[i] = data[i][:, sensor_idx, :]

    return data, labels, subjects, label_info, subject_list


def generate_training_test_data(data, label, subjects, subject_list, training_portion=0.7, shuffle=False, cv=-1, n_cv=1, verbose=0):

    training_size = int(len(subject_list)*training_portion)

    sub_list = list(subject_list.copy())
    if shuffle:
        np.random.shuffle(sub_list)

    if cv > 0:
        test_size = int(len(subject_list) * 1/cv)

        idx1 = int(len(subject_list) * (1-(n_cv/cv)))
        idx2 = min(idx1+test_size, len(subject_list))

        training_subjects = sub_list[0:idx1] + sub_list[idx2:]
    else:
        training_subjects = sub_list[:training_size]

    if verbose > 0:
        for sub in subject_list:
            if sub in training_subjects:
                print(" {} , ".format(sub), end="")
            else:
                print("|{}|, ".format(sub), end="")
        print("")

    training_data = []
    training_label = []
    training_subject = []
    test_data = []
    test_label = []
    test_subject = []

    for i in range(len(data)):
        if subjects[i] in training_subjects:
            training_data.append(data[i])
            training_label.append(label[i])
            training_subject.append(subjects[i])
        else:
            test_data.append(data[i])
            test_label.append(label[i])
            test_subject.append(subjects[i])

    return training_data, training_label, training_subject, test_data, test_label, test_subject


def make_training_data(data_root, save_root=None, window_size=300, stride=1, chunk_size=50,
                       normalize_axis=True, normalize_max=1,
                       merge_clap_null=True, training_portion=0.8, shuffle=True,
                       verbose=1):
    """

    :param data_root: Root directory of dataset
    :param save_root: Root directory of generated data. None if window_size < 1
    :param window_size: Less than 1 will return without sliding windowed data as (training_data, training_label, training_subject, test_data, test_label, test_subject)
    :param stride: Size of stride
    :param chunk_size:
    :param normalize_axis: Normalize each axis from 0 to normalize_max by calculating each axis's min/max values
    :param normalize_max: Maximum value to be set when normalize_axis is True
    :param merge_clap_null: Treats clap label as null
    :param training_portion: Percentage of subjects included in training data from entire dataset
    :param shuffle: Shuffle subjects order.
    :param verbose:
    :return:
    """
    assert (save_root is not None and window_size > 0) or window_size < 1

    data, label, subjects, label_info, subject_list = prepare_data(data_root, merge_clap_null=merge_clap_null)

    if normalize_axis:
        data = normalize_data(data, norm_max=normalize_max)

    training_data, training_label, training_subject, test_data, test_label, test_subject = \
        generate_training_test_data(data, label, subjects, subject_list, training_portion=training_portion,
                                    shuffle=shuffle, verbose=verbose)

    if window_size < 1:
        return training_data, training_label, training_subject, test_data, test_label, test_subject

    save_windowed_dataset_hdf5(training_data, training_label, test_data, test_label,
                               window_size=window_size, stride=stride, save_root=save_root,
                               chunk_size=chunk_size)

    with open("{}label_info.json".format(save_root), "w") as f:
        json.dump(label_info, f)


if __name__ == "__main__":
    pass
