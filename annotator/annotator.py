# -*- coding: utf8 -*-

try:
    from ..utils.data_loader import read_mat_file, read_csv_file
except:
    from utils.data_loader import read_mat_file, read_csv_file

import platform
import cv2
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import numpy as np
import argparse
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class LabelMakerApp:
    def __init__(self, window, video_root, csv_root=None, mat_root=None, v_width=640, title="Label Maker"):
        self.window = window

        self.video_root = video_root
        self.csv_root = csv_root

        # Reading video and data START
        cap = cv2.VideoCapture(video_root)

        height = int((v_width / int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (height, v_width)

        self.frame_size = frame_size
        self.vcap = cap
        self.video_duration = self.vcap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vcap.get(cv2.CAP_PROP_FPS)
        self.max_frame_count = self.vcap.get(cv2.CAP_PROP_FRAME_COUNT)

        self.segment_info = dict()
        self.segment_info['x'] = []
        self.segment_info['data'] = []
        self.segment_info['label'] = []
        self.segment_info['name'] = []

        self.latest_slide = 0
        self.slider_change = False
        self.segment_change = False
        self.play_segment = False
        self.play_segment_idx = 0

        self.video_sync_ms = 0

        if csv_root is not None:
            sensor_data, duration = read_csv_file(csv_root)
            self.sensor_data = sensor_data
            self.duration = duration
        elif mat_root is not None:
            csv_data, vid_data, segment_info = read_mat_file(mat_root)

            self.sensor_data = np.concatenate([np.tile(csv_data['x'], (5, 1, 1)).swapaxes(0, 2), csv_data['sensor_data']], axis=1).swapaxes(0, 2).swapaxes(1, 2)
            self.duration = csv_data['duration']

            self.segment_info['x'] = segment_info['segment_x'].tolist()
            self.segment_info['data'] = segment_info['segment_sensor_data'].tolist()
            self.segment_info['label'] = segment_info['segment_label'].tolist()
            self.segment_info['name'] = segment_info['segment_name'].tolist()
            self.video_sync_ms = segment_info['video_sync_time'] * 1000
            self.csv_root = csv_data['file_path']
        # Reading video and data END

        self.delay = 10

        # 01. Create video preview window
        frame_canvas = tk.Frame(window)
        self.canvas = tk.Canvas(frame_canvas, width=self.frame_size[1], height=self.frame_size[0])
        self.canvas.pack(side=tk.LEFT, padx=10)
        frame_canvas.pack()

        # 03. Create Buttons START
        frame_btn = tk.Frame(frame_canvas)
        self.btn_play_segment = tk.Button(frame_btn, text="Play Segment", command=self.button_play_segment, height=2, width=15)
        self.btn_play_segment.pack(side=tk.TOP, pady=5)

        self.btn_gen_segment = tk.Button(frame_btn, text="Generate Segment", command=self.button_gen_segment, height=2, width=15)
        self.btn_gen_segment.pack(side=tk.TOP, pady=5)

        self.btn_gen_segment = tk.Button(frame_btn, text="Delete Segment", command=self.button_del_segment, height=2, width=15)
        self.btn_gen_segment.pack(side=tk.TOP, pady=5)

        self.btn_save = tk.Button(frame_btn, text="Save", command=self.button_save, height=2, width=15)
        self.btn_save.pack(side=tk.TOP, pady=5)

        self.btn_zoom_reset_figure = tk.Button(frame_btn, text="Zoom reset", command=self.button_zoom_reset_figure,
                                            height=1, width=15)
        self.btn_zoom_reset_figure.pack(side=tk.TOP, pady=15)
        self.btn_zoom_in_figure = tk.Button(frame_btn, text="Zoom +", command=self.button_zoom_in_figure, height=1, width=15)
        self.btn_zoom_in_figure.pack(side=tk.TOP, pady=0)
        self.btn_zoom_out_figure = tk.Button(frame_btn, text="Zoom -", command=self.button_zoom_out_figure, height=1, width=15)
        self.btn_zoom_out_figure.pack(side=tk.TOP, pady=0)

        frame_btn.pack(side=tk.RIGHT, padx=20)
        # 03. Create Buttons END

        # 04. Create Label and class name inputs START
        frame_label_and_options = tk.Frame(frame_canvas)
        tk.Label(frame_label_and_options, text="Label").pack()
        self.entry_label = tk.Entry(frame_label_and_options)
        self.entry_label.delete(0, tk.END)
        self.entry_label.insert(0, "0")
        self.entry_label.pack()

        tk.Label(frame_label_and_options, text="Name").pack()
        self.entry_name = tk.Entry(frame_label_and_options)
        self.entry_name.delete(0, tk.END)
        self.entry_name.insert(0, "Name")
        self.entry_name.pack()

        tk.Label(frame_label_and_options, text="Segmentation list").pack()

        frame01 = tk.Frame(frame_label_and_options)
        scrollbar = tk.Scrollbar(frame01, orient="vertical")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox_segments = tk.Listbox(frame01, yscrollcommand=scrollbar.set)
        self.listbox_segments.pack(side=tk.LEFT, fill=tk.Y)
        self.listbox_segments.bind('<<ListboxSelect>>', self.on_listbox_selected)

        scrollbar["command"] = self.listbox_segments.yview

        frame01.pack()
        # 04. Create Label and class name inputs END

        # 05. Create sensor index selection START
        tk.Label(frame_label_and_options, text="Sensor Index").pack()
        frame02 = tk.Frame(frame_label_and_options)
        self.sensor_idx = tk.StringVar()
        tk.Radiobutton(frame02, text="01", value=0, var=self.sensor_idx, command=self.radio_button_sensor_idx).pack(side=tk.LEFT)
        tk.Radiobutton(frame02, text="02", value=1, var=self.sensor_idx, command=self.radio_button_sensor_idx).pack(side=tk.LEFT)
        tk.Radiobutton(frame02, text="03", value=2, var=self.sensor_idx, command=self.radio_button_sensor_idx).pack(side=tk.LEFT)
        tk.Radiobutton(frame02, text="04", value=3, var=self.sensor_idx, command=self.radio_button_sensor_idx).pack(side=tk.LEFT)
        tk.Radiobutton(frame02, text="05", value=4, var=self.sensor_idx, command=self.radio_button_sensor_idx).pack(side=tk.LEFT)
        frame02.pack()
        self.sensor_idx.set(0)
        # 05. Create sensor index selection END

        frame_label_and_options.pack(side=tk.RIGHT)

        # 02. Create slider for video sync and segmentation selection START
        frame_label_and_options.update()
        frame_btn.update()
        slider_width = (self.frame_size[1] + frame_label_and_options.winfo_width() + frame_btn.winfo_width()) * 0.95

        frame_sliders = tk.Frame(window)
        self.slider_sync = tk.Scale(frame_sliders, from_=-self.max_frame_count, to=self.max_frame_count,
                                    length=slider_width, orient=tk.HORIZONTAL, command=self.slider_sync_set)
        self.slider_sync.set(int(((self.video_sync_ms / 1000) / self.video_duration) * self.max_frame_count))
        self.slider_sync.pack()

        self.slider_sensor_s = tk.Scale(frame_sliders, from_=0, to=self.max_frame_count * 1.1,
                                        length=slider_width,
                                        orient=tk.HORIZONTAL, command=self.slider_set)
        self.slider_sensor_s.set(0)
        self.slider_sensor_s.pack()
        self.slider_sensor_e = tk.Scale(frame_sliders, from_=0, to=self.max_frame_count * 1.1,
                                        length=slider_width,
                                        orient=tk.HORIZONTAL, command=self.slider_set)
        self.slider_sensor_e.set(0)
        self.slider_sensor_e.pack()
        frame_sliders.pack(side=tk.BOTTOM, pady=10)
        # 02. Create slider for video sync and segmentation selection END

        # 06. Create sensor data plot figure START
        f_w = self.frame_size[1] / 120
        f_h = self.frame_size[0] / 146

        self.figure = figure.Figure(figsize=(f_w, f_h))
        self.axes = dict()
        self.lines = dict()
        self.line_s = dict()
        self.line_e = dict()

        self.axes['acc'] = self.figure.add_subplot(311)
        self.lines['acc'] = self.axes['acc'].plot(self.sensor_data[0][:, 0], self.sensor_data[0][:, 1:4])
        self.axes['acc'].set_xticks([])

        self.line_s['acc'] = self.axes['acc'].plot([1, 1], [self.sensor_data[0][:, 1:4].min(), self.sensor_data[0][:, 1:4].max()], color='r', alpha=0.7)
        self.line_e['acc'] = self.axes['acc'].plot([20, 20], [self.sensor_data[0][:, 1:4].min(), self.sensor_data[0][:, 1:4].max()], color='b', alpha=0.7)

        self.axes['gyro'] = self.figure.add_subplot(312)
        self.lines['gyro'] = self.axes['gyro'].plot(self.sensor_data[0][:, 0], self.sensor_data[0][:, 4:7])
        self.axes['gyro'].set_xticks([])

        self.line_s['gyro'] = self.axes['gyro'].plot([1, 1], [self.sensor_data[0][:, 4:7].min(), self.sensor_data[0][:, 4:7].max()], color='r', alpha=0.7)
        self.line_e['gyro'] = self.axes['gyro'].plot([20, 20], [self.sensor_data[0][:, 4:7].min(), self.sensor_data[0][:, 4:7].max()], color='b', alpha=0.7)

        self.axes['ori'] = self.figure.add_subplot(313)
        self.lines['ori'] = self.axes['ori'].plot(self.sensor_data[0][:, 0], self.sensor_data[0][:, 7:])

        self.line_s['ori'] = self.axes['ori'].plot([1, 1], [self.sensor_data[0][:, 7:].min(), self.sensor_data[0][:, 7:].max()], color='r', alpha=0.7)
        self.line_e['ori'] = self.axes['ori'].plot([20, 20], [self.sensor_data[0][:, 7:].min(), self.sensor_data[0][:, 7:].max()], color='b', alpha=0.7)

        self.figure_canvas = FigureCanvasTkAgg(self.figure, master=window)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack(side=tk.RIGHT)
        # 06. Create sensor data plot figure END

        # 07. Create segmented sensor data plot figure START
        self.figure_zoom = figure.Figure(figsize=(f_w, f_h))
        self.axes_zoom = dict()
        self.lines_zoom = dict()
        self.line_zoom_play = dict()

        self.axes_zoom['acc'] = self.figure_zoom.add_subplot(311)
        self.lines_zoom['acc'] = self.axes_zoom['acc'].plot(self.sensor_data[0][0:0, 1:4])
        self.line_zoom_play['acc'] = self.axes_zoom['acc'].plot([0, 0], [0, 0], 'r')
        self.axes_zoom['acc'].set_xticks([])

        self.axes_zoom['gyro'] = self.figure_zoom.add_subplot(312)
        self.lines_zoom['gyro'] = self.axes_zoom['gyro'].plot(self.sensor_data[0][0:0, 1:4])
        self.line_zoom_play['gyro'] = self.axes_zoom['gyro'].plot([0, 0], [0, 0], 'r')
        self.axes_zoom['gyro'].set_xticks([])

        self.axes_zoom['ori'] = self.figure_zoom.add_subplot(313)
        self.lines_zoom['ori'] = self.axes_zoom['ori'].plot(self.sensor_data[0][0:0, 1:4])
        self.line_zoom_play['ori'] = self.axes_zoom['ori'].plot([0, 0], [0, 0], 'r')

        self.figure_canvas_zoom = FigureCanvasTkAgg(self.figure_zoom, master=window)
        self.figure_canvas_zoom.draw()
        self.figure_canvas_zoom.get_tk_widget().pack()
        # 07. Create segmented sensor data plot figure END

        self.s_slide_idx = 0
        self.e_slide_idx = 0

        self.photo = None
        self.axis_change = False
        self.zoom_ratio = 20

        if mat_root is not None:
            self.update_listbox_segment()

            self.segment_change = True

            self.line_update('acc')
            self.line_update('gyro')
            self.line_update('ori')

            self.figure_canvas.draw()
            self.figure_canvas_zoom.draw()

        self.update()
        self.window.mainloop()

    def on_listbox_selected(self, event):
        if len(event.widget.curselection()) != 1:
            return
        else:
            idx = event.widget.curselection()[0]

        s_idx = self.segment_info['x'][idx][0]
        e_idx = self.segment_info['x'][idx][-1]

        s_slide_idx = int(np.ceil((s_idx / self.video_duration) * self.max_frame_count))
        e_slide_idx = int(np.ceil((e_idx / self.video_duration) * self.max_frame_count))

        self.slider_sensor_s.set(s_slide_idx)
        self.slider_sensor_e.set(e_slide_idx)

        self.slider_change = True

    def update_listbox_segment(self):
        self.listbox_segments.delete(0, tk.END)
        for x, label, name in zip(self.segment_info['x'], self.segment_info['label'], self.segment_info['name']):
            self.listbox_segments.insert(tk.END, "%06.1fs, [%02d] %s" % (x[0], label, name))

    def radio_button_sensor_idx(self):
        self.axis_change = True

    def button_gen_segment(self):
        segmented_sensor = [self.sensor_data[i][self.s_slide_idx:self.e_slide_idx + 1, 1:] for i in range(len(self.sensor_data))]
        segmented_x = self.sensor_data[0][self.s_slide_idx:self.e_slide_idx + 1, 0]
        segmented_sensor = np.swapaxes(np.array(segmented_sensor), 0, 1)
        segmented_sensor = np.swapaxes(segmented_sensor, 1, 2)

        self.segment_info['x'].append(segmented_x)
        self.segment_info['data'].append(segmented_sensor)
        self.segment_info['label'].append(int(self.entry_label.get()))
        self.segment_info['name'].append(str(self.entry_name.get()))

        self.update_listbox_segment()

        self.slider_change = True
        self.segment_change = True

    def button_del_segment(self):
        if len(self.listbox_segments.curselection()) != 1:
            return
        else:
            idx = self.listbox_segments.curselection()[0]

        self.listbox_segments.delete(idx)

        del self.segment_info['x'][idx]
        del self.segment_info['data'][idx]
        del self.segment_info['label'][idx]
        del self.segment_info['name'][idx]

        self.slider_change = True
        self.segment_change = True

    def button_save(self):
        csv_file = dict()
        csv_file['path'] = self.csv_root
        for i in range(5):
            csv_file['data%d'% i] = self.sensor_data[i][:, 1:]
        csv_file['x'] = self.sensor_data[0][:, 0]

        vid_file = dict()
        vid_file['path'] = self.video_root

        self.segment_info['video_sync_time'] = self.video_sync_ms / 1000
        m_list = np.zeros((len(self.segment_info['name']), ), dtype=np.object)
        m_list[:] = self.segment_info['name']
        self.segment_info['name'] = m_list

        save_mat = dict()
        save_mat['csv_file'] = csv_file
        save_mat['vid_file'] = vid_file
        save_mat['segment_info'] = self.segment_info
        save_mat['version'] = 'python'
        path = filedialog.asksaveasfilename(initialdir='./')

        savemat(path, save_mat, appendmat=False, do_compression=True)

    def button_play_segment(self):
        if self.slider_sensor_s.get() >= self.slider_sensor_e.get():
            return

        self.play_segment = not self.play_segment
        self.play_segment_idx = self.slider_sensor_s.get()
        self.set_play_segment()

    def button_zoom_in_figure(self):
        self.zoom_ratio *= 0.7
        self.slider_change = True

    def button_zoom_out_figure(self):
        self.zoom_ratio = min(self.zoom_ratio / 0.7, 20)
        self.slider_change = True

    def button_zoom_reset_figure(self):
        if self.zoom_ratio >= 20:
            return
        else:
            self.zoom_ratio = 20
            self.slider_change = True

    def set_play_segment(self):
        if self.play_segment:
            self.btn_play_segment['text'] = "Stop Segment"
        else:
            self.btn_play_segment['text'] = "Play Segment"

    def slider_set(self, value):
        self.slider_change = True
        self.latest_slide = int(value)

    def slider_sync_set(self, value):
        self.slider_change = True

    def line_update(self, sensor_type, sensor_idx=0):
        if sensor_type is 'acc':
            idx_adder = 1
        elif sensor_type is 'gyro':
            idx_adder = 4
        else:
            idx_adder = 7

        if self.axis_change:
            for i, line in enumerate(self.lines[sensor_type]):
                line.set_xdata(self.sensor_data[sensor_idx][:, 0])
                line.set_ydata(self.sensor_data[sensor_idx][:, i+idx_adder])
            max_y = self.sensor_data[sensor_idx][:, idx_adder:idx_adder+3].max()
            min_y = self.sensor_data[sensor_idx][:, idx_adder:idx_adder + 3].min()

            self.line_s[sensor_type][0].set_ydata([min_y, max_y])
            self.line_e[sensor_type][0].set_ydata([min_y, max_y])
            self.axes[sensor_type].set(ylim=(min_y, max_y))

        self.line_s[sensor_type][0].set_xdata([self.sensor_data[sensor_idx][self.s_slide_idx, 0], self.sensor_data[sensor_idx][self.s_slide_idx, 0]])
        self.line_e[sensor_type][0].set_xdata([self.sensor_data[sensor_idx][self.e_slide_idx, 0], self.sensor_data[sensor_idx][self.e_slide_idx, 0]])

        for i, line in enumerate(self.lines_zoom[sensor_type]):
            line.set_xdata(self.sensor_data[sensor_idx][self.s_slide_idx:self.e_slide_idx + 1, 0])
            line.set_ydata(self.sensor_data[sensor_idx][self.s_slide_idx:self.e_slide_idx + 1, i + idx_adder])

        self.axes_zoom[sensor_type].set(xlim=(self.sensor_data[sensor_idx][self.s_slide_idx, 0],
                                              self.sensor_data[sensor_idx][self.e_slide_idx, 0]),
                                        ylim=(self.sensor_data[sensor_idx][self.s_slide_idx:self.e_slide_idx + 1, idx_adder:idx_adder+3].min(),
                                              self.sensor_data[sensor_idx][self.s_slide_idx:self.e_slide_idx + 1, idx_adder:idx_adder+3].max()))

        if self.segment_change or self.axis_change:
            remove_lines = []
            for i in range(5, len(self.axes[sensor_type].get_lines())):
                remove_lines.append(self.axes[sensor_type].get_lines()[i])

            for r_line in remove_lines:
                r_line.remove()

            for text in self.axes[sensor_type].texts:
                text.remove()

            for i in range(len(self.segment_info['x'])):
                x = self.segment_info['x'][i]
                y = self.segment_info['data'][i][:, idx_adder-1:idx_adder+2, sensor_idx]
                label = str(self.segment_info['label'][i])

                self.axes[sensor_type].plot(x, y, 'k', alpha=0.5)
                text_location = i % 6

                if text_location < 3:
                    self.axes[sensor_type].text(x[0], y[:, text_location].max(), label, fontsize=8, alpha=0.8,
                                                verticalalignment='top', bbox=dict(boxstyle="square", fc=(1, 1, 1, 0.3),
                                                                                   ec=(1, 1, 1, 0)))
                else:
                    self.axes[sensor_type].text(x[0], y[:, text_location-3].min(), label, fontsize=8, alpha=0.8,
                                                verticalalignment='top', bbox=dict(boxstyle="square", fc=(1, 1, 1, 0.3),
                                                                                   ec=(1, 1, 1, 0)))
            if sensor_type is 'ori':
                self.segment_change = False

        if self.zoom_ratio < 20:
            zoom_width = (self.e_slide_idx - self.s_slide_idx)
            idx_s = int(max(self.s_slide_idx - zoom_width*self.zoom_ratio+1, 0))
            idx_e = int(min(self.e_slide_idx + zoom_width*self.zoom_ratio+1, self.sensor_data[sensor_idx].shape[0]-1))

            self.axes[sensor_type].set(xlim=(self.sensor_data[sensor_idx][idx_s, 0],
                                                  self.sensor_data[sensor_idx][idx_e, 0]),
                                            ylim=(self.sensor_data[sensor_idx][idx_s:idx_e + 1,
                                                  idx_adder:idx_adder + 3].min(),
                                                  self.sensor_data[sensor_idx][idx_s:idx_e + 1,
                                                  idx_adder:idx_adder + 3].max()))
        else:
            self.axes[sensor_type].set(xlim=(self.sensor_data[sensor_idx][0, 0],
                                             self.sensor_data[sensor_idx][-1, 0]),
                                       ylim=(self.sensor_data[sensor_idx][:,idx_adder:idx_adder + 3].min(),
                                             self.sensor_data[sensor_idx][:,idx_adder:idx_adder + 3].max()))

    def se_slider_update(self, s_slide, e_slide, sensor_idx=0):
        self.s_slide_idx = int(np.argmax(self.sensor_data[sensor_idx][:, 0] > s_slide) - 1)
        self.e_slide_idx = int(np.argmax(self.sensor_data[sensor_idx][:, 0] > e_slide) - 1)

        if self.e_slide_idx >= self.sensor_data[sensor_idx].shape[0] or self.e_slide_idx < 0:
            self.e_slide_idx = self.sensor_data[sensor_idx].shape[0] - 1

        if self.s_slide_idx < 0:
            self.s_slide_idx = 0

    def play_line_update(self, key_frame_ms=-1):
        for s_type in ['acc', 'gyro', 'ori']:
            if key_frame_ms < 0:
                self.line_zoom_play[s_type][0].set_xdata([0, 0])
                self.line_zoom_play[s_type][0].set_ydata([0, 0])
                continue

            if s_type is 'acc':
                idx_adder = 1
            elif s_type is 'gyro':
                idx_adder = 4
            else:
                idx_adder = 7

            sensor_idx = int(self.sensor_idx.get())

            p_idx = int(np.argmax(self.sensor_data[sensor_idx][:, 0] > (key_frame_ms/1000)) - 1)

            play_x = self.sensor_data[sensor_idx][p_idx, 0]
            min_y = self.sensor_data[sensor_idx][self.s_slide_idx:self.e_slide_idx + 1, idx_adder:idx_adder + 3].min()
            max_y = self.sensor_data[sensor_idx][self.s_slide_idx:self.e_slide_idx + 1, idx_adder:idx_adder + 3].max()

            self.line_zoom_play[s_type][0].set_xdata([play_x, play_x])
            self.line_zoom_play[s_type][0].set_ydata([min_y, max_y])

        self.figure_canvas_zoom.draw()

    def update(self):
        self.video_sync_ms = ((self.slider_sync.get() / self.max_frame_count) * self.video_duration) * 1000

        if self.play_segment:
            key_frame_ms = ((self.play_segment_idx / self.max_frame_count) * self.video_duration) * 1000

            self.play_line_update(key_frame_ms=key_frame_ms)

            self.play_segment_idx += 1
            if self.play_segment_idx > self.slider_sensor_e.get():
                self.play_segment = False
                self.set_play_segment()
                self.play_line_update(key_frame_ms=-1)
        else:
            key_frame_ms = ((self.latest_slide / self.max_frame_count) * self.video_duration) * 1000

        self.vcap.set(cv2.CAP_PROP_POS_MSEC, key_frame_ms + self.video_sync_ms)

        ret, frame = self.vcap.read()

        if ret:
            frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        if self.slider_change or self.axis_change:
            s_slide = (self.slider_sensor_s.get() / self.max_frame_count) * self.video_duration
            e_slide = (self.slider_sensor_e.get() / self.max_frame_count) * self.video_duration

            if s_slide < e_slide:
                self.se_slider_update(s_slide, e_slide, sensor_idx=int(self.sensor_idx.get()))

                self.line_update('acc', sensor_idx=int(self.sensor_idx.get()))
                self.line_update('gyro', sensor_idx=int(self.sensor_idx.get()))
                self.line_update('ori', sensor_idx=int(self.sensor_idx.get()))

                self.figure_canvas.draw()
                self.figure_canvas_zoom.draw()

            self.slider_change = False
            self.axis_change = False

        self.window.after(self.delay, self.update)

    def _quit(self):
        self.window.quit()
        self.window.destroy()


def get_file_name_from_path(path):
    # TODO : it looks like windows path also uses '/'
    if platform.system() == "Windows":
        seperator = "\\"
    else:
        seperator = "/"

    idx = path.find(seperator)
    while idx >= 0:
        path = path[idx + 1:]
        idx = path.find(seperator)
    return path


def get_file_path(root_, initial_dir ='./'):

    video_path = filedialog.askopenfilename(initialdir=initial_dir, title="Select the video file", parent=root_)
    data_path = filedialog.askopenfilename(initialdir=initial_dir, title="Select the csv or mat file",
                                           filetypes=[("CSV file", ".csv"),
                                                       ("Matlab file", ".mat")], parent=root_)
    return video_path, data_path


def run_annotator(video_path_, data_path_, v_width=640):
    root = tk.Tk()

    if video_path_ == "" or data_path_ == "":
        video_path_, data_path_ = get_file_path(root)

    if video_path_ is "" or data_path_ is "":
        root.quit()
        root.destroy()
        exit()

    data_name = get_file_name_from_path(data_path_)
    video_name = get_file_name_from_path(video_path_)

    root.title("Data Label Maker - {} <-> {}".format(video_name, data_name))

    if data_path_.endswith("csv"):
        LabelMakerApp(root, video_path_, csv_root=data_path_, v_width=v_width)
    elif data_path_.endswith("mat"):
        LabelMakerApp(root, video_path_, mat_root=data_path_, v_width=v_width)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Label Maker")
    parser.add_argument('--vw', type=int, default=640, help="Video width to be displayed. Minimum : 640")
    parser.add_argument('--video-path', type=str, default="", help="Video file path for labeling")
    parser.add_argument('--data-path', type=str, default="", help="Data file path(csv or mat) for labeling")
    args = parser.parse_args()

    run_annotator(args.video_path, args.data_path, v_width=args.vw)
