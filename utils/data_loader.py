import scipy.io as sio
import numpy as np
import pandas as pd


def read_mat_file(path):
    mat = sio.loadmat(path)

    if 'version' in mat.keys() and 'python' in mat['version']:
            is_from_python = True
    else:
        is_from_python = False

    # Read csv file data START
    if is_from_python:
        file_path = mat['csv_file'][0][0][0][0]
        file_name = file_path
        sensor01 = mat['csv_file']['data0'][0][0]
        sensor02 = mat['csv_file']['data1'][0][0]
        sensor03 = mat['csv_file']['data2'][0][0]
        sensor04 = mat['csv_file']['data3'][0][0]
        sensor05 = mat['csv_file']['data4'][0][0]

        x = mat['csv_file']['x'][0][0][0]
        duration = x[-1] - x[0]
        original_data = None
    else:
        file_name = mat['csv_file'][0][0][0]
        file_path = mat['csv_file'][0][0][1]

        original_data = np.array(mat['csv_file'][0][0][2])
        duration = mat['csv_file'][0][0][3][0][0]
        sensor01 = np.array(mat['csv_file'][0][0][4])
        sensor02 = np.array(mat['csv_file'][0][0][5])
        sensor03 = np.array(mat['csv_file'][0][0][6])
        sensor04 = np.array(mat['csv_file'][0][0][7])
        sensor05 = np.array(mat['csv_file'][0][0][8])

        x = np.array(mat['csv_file'][0][0][9]).flatten()

    csv_data = dict()
    csv_data["file_name"] = file_name
    csv_data["file_path"] = file_path
    csv_data["original_data"] = original_data
    csv_data["duration"] = duration
    csv_data["sensor_data"] = np.dstack((sensor01, sensor02, sensor03, sensor04, sensor05))

    for i in range(csv_data["sensor_data"].shape[2]):
        mask = np.isnan(csv_data["sensor_data"][:, :, i])
        mask_idx = np.argwhere(mask)
        for idx in mask_idx:
            idx1 = idx[0]
            idx2 = idx[1]
            csv_data["sensor_data"][idx1, idx2, i] = csv_data["sensor_data"][idx1-1, idx2, i]

    csv_data["x"] = x
    # Read csv file data END

    # Read video information START
    if is_from_python:
        vid_path = mat['vid_file'][0][0][0][0]
        vid_name = vid_path
    else:
        vid_name = mat['vid_file'][0][0][0]
        vid_path = mat['vid_file'][0][0][1]

    vid_data = dict()
    vid_data["vid_name"] = vid_name
    vid_data["vid_path"] = vid_path
    # Read video information START

    # Read segmentation data START
    if is_from_python:
        segment_x = mat['segment_info']['x'][0][0][0]
        segment_sensor_data = mat['segment_info']['data'][0][0][0]
        segment_label = mat['segment_info']['label'][0][0][0]
        segment_name = mat['segment_info']['name'][0][0][0]
        video_sync_time = mat['segment_info']['video_sync_time'][0][0][0][0]
    else:
        video_sync_time = mat['segment_info'][0][0][0][0][0]
        segment_x = np.array(mat['segment_info'][0][0][1][0])
        segment_sensor_data = np.array(mat['segment_info'][0][0][2][0])
        segment_label = np.array(mat['segment_info'][0][0][3][0])
        segment_name = np.array(mat['segment_info'][0][0][4][0])

    for i in range(segment_x.shape[0]):
        segment_x[i] = segment_x[i].flatten()

    # Fixing NaN value on sensor segments
    for i in range(segment_sensor_data.shape[0]):
        for j in range(segment_sensor_data[i].shape[2]):
            mask = np.isnan(segment_sensor_data[i][:, :, j])
            mask_idx = np.argwhere(mask)
            for idx in mask_idx:
                idx1, idx2 = idx[0], idx[1]
                segment_sensor_data[i][idx1, idx2, j] = segment_sensor_data[i][idx1-1, idx2, j]

    segment_data = dict()
    segment_data["video_sync_time"] = video_sync_time
    segment_data["segment_x"] = segment_x
    segment_data["segment_sensor_data"] = segment_sensor_data
    segment_data["segment_label"] = segment_label
    segment_data["segment_name"] = segment_name

    for i in range(len(segment_data['segment_name'])):
        segment_data['segment_name'][i] = segment_data['segment_name'][i][0]

    return csv_data, vid_data, segment_data


def read_csv_file(path, video_duration=None):
    keys = ["SensorIndex", "Timestamp", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "oriX", "oriY", "oriZ"]
    csv_file = pd.read_csv(path, names=keys)

    duration = (csv_file['Timestamp'].loc[csv_file.shape[0]-1] - csv_file["Timestamp"].loc[0]) / 1000

    unique_idx = csv_file['SensorIndex'].unique()
    unique_idx.sort()

    if unique_idx.shape[0] != 5:
        print("Number of sensors must be 5!")
        return

    sensor_data = []

    for idx in unique_idx:
        s_data = csv_file.query("SensorIndex == %d" % idx)[keys[1:]].copy(deep=True)

        s_data.drop_duplicates("Timestamp", inplace=True)
        s_data.reset_index(inplace=True)

        s_data["Timestamp"] = (s_data["Timestamp"] - s_data["Timestamp"].loc[0]) / 1000

        sensor_data.append(s_data)

    max_length = max([sd.shape[0] for sd in sensor_data])

    x = np.linspace(0, duration, max_length)

    n_sensor_data = []
    for s_data in sensor_data:
        d_frame = pd.DataFrame()
        for key in s_data.keys():
            d_frame[key] = np.interp(x, s_data['Timestamp'], s_data[key])
        d_frame.drop(columns="index", inplace=True)
        n_sensor_data.append(d_frame.values)

    if video_duration is not None:
        for n_data in n_sensor_data:
            n_data[:, 0] = (n_data[:, 0] / n_data[:, 0].max()) * video_duration

    return n_sensor_data, duration
