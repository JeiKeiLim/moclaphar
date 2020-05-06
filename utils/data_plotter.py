import matplotlib.pyplot as plt
import numpy as np


def draw_segmentation(csv_data, segment_data, sensor_idx=range(0,3), figure_size=(20, 10)):
    plt.figure(figsize=figure_size)
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(csv_data["x"], csv_data["sensor_data"][:, sensor_idx, i])

    text_up = True
    text_y_basis = 0
    for i in range(segment_data["segment_x"].shape[0]):
        for j in range(5):
            plt.subplot(5, 1, j + 1)
            plt.plot(segment_data["segment_x"][i], segment_data["segment_sensor_data"][i][:, sensor_idx, j])

            sensor_max = np.nanmax(csv_data["sensor_data"][:, sensor_idx, j])
            sensor_min = np.nanmin(csv_data["sensor_data"][:, sensor_idx, j])

            text_x = segment_data["segment_x"][i][0]
            if text_up:
                text_up = False
                text_y = sensor_max - text_y_basis
            else:
                text_up = True
                text_y = sensor_min + text_y_basis

                if text_y_basis <= 0:
                    text_y_basis = 5
                else:
                    text_y_basis = 0

            plt.text(text_x, text_y, segment_data["segment_name"][i])

    plt.show()
