# 1. moclaphar (v0.0.30)
Motion Classification Human Activity Recognition Helper Package


# 2. Getting started
## 2.1 Environments
- Python 3.6

## 2.2 Install package with pip install
```
pip install moclaphar
```

# 3. Structure
## 3.1 moclaphar.annotator
### 3.1.1 Running annotator
* Run from command line with data path given
```python ./annotator/annotator.py --vw 640 --video-path <video path> --data-path <data path> ``` 

* Run from command line without data path given ```python ./annotator/annotator.py --vw 640```
  - It prompts file selector asking where the video and data are.
  
* Run from python script with data path given
```python
from moclaphar.annotator import run_annotator

run_annotator("/video/file/path/", "/data/file/path/csv/or/mat/", vw=1024)
```

* Run from python script without data path given
```python
from moclaphar.annotator import run_annotator

run_annotator("", "", vw=1024)
```
The file selector dialogue will prompt.


## 3.2 moclaphar.dataset
### 3.2.1 moclaphar.dataset.dataset.py
* get_file_list(root, ext="")
  - Find files with ```ext``` extension recursively and returns its file paths.
  - Arguments
    - root: root directory of data path
    - ext: extension of target file.
  - Return: list of file paths with matching extension.
  
* prepare_data(root, accelerometer=True, gyroscope=True, orientation=False, stroke=False, merge_clap_null=True, verbose=1)
  - Read .mat files from root directory and returns data
   
  - Arguments
    - root: Root directory of data path
    - accelerometer: True includes accelerometer sensor data.
    - gyroscope: True includes gyroscope sensor data.
    - orientation: True includes orientation data.
    - merge_clap_null: True treats clap label as null.
    - verbose: 1 shows label histogram information. 0 silence.
  - Return
    - data(list): List of activity segmented sensor data.
    - labels(list): Ground truth of each activity. Label id number might differ from original label ids since it recreates the number. 
    - subjects(list): Name of subject of each activity.
    - label_info(dict): Contains matching class name and id.
    - subject_list(list): Contains names of subject included in return data. This list is sorted by alphabet order.

* generate_training_test_data(data, label, subjects, subject_list, training_portion=0.7, shuffle=False, cv=-1, n_cv=1, verbose=0)
  - Split data into training and test set.
  - Arguments
    - data(list): List of activity segmented sensor data.
    - label(list): Ground truth of each activity. Label id number might differ from original label ids since it recreates the number.
    - subject(list): Name of subject of each activity.
    - subject_list(list): Contains names of subject included in return data. This list is sorted by alphabet order.
    - training_portion(float): portion of training data. 0.7 will split data into 70% subjects in training set and 30% subjects in test set.
    - shuffle: True shuffles subject_list.
    - cv: Cross validation number. It only applies when cv is greater than 0.
    - n_cv: Number of cross-validation iteration. Ex) cv=5, n_cv=5. 5 cross validation of fifth iteration. 
    - verbose: 1 shows training and test subject names. 0 silence.
   - Return
     - training_data: Activity segmented sensor data.
     - training_label: Ground truth of each segmented activity.
     - training_subject: Name of subject of each segmented activity.
     - test_data: Activity segmented sensor data.
     - test_label: Ground truth of each segmented activity.
     - test_subject  Name of subject of each segmented activity.
     
* make_training_data
  - Generate sliding windowed training and test dataset from data_root and save hdf5 file into save_root or returns training and test data without sliding window
  - Argument
    - data_root: Root directory of dataset containing .mat
    - save_root: Root directory of generated data. None if window_size < 1
    - window_size: Less than 1 will return without sliding windowed data as (training_data, training_label, training_subject, test_data, test_label, test_subject)
    - stride: Size of stride
    - chunk_size: Amount of data processing when writing slide windowed data into hdf5. Recommended to set low value for PC with low memory
    - normalize_axis: Normalize each axis from 0 to normalize_max by calculating each axis's min/max values
    - normalize_max: Maximum value to be set when normalize_axis is True
    - merge_clap_null: Treats clap label as null
    - training_portion: Percentage of subjects included in training data from entire dataset
    - shuffle: Shuffle subjects order.
    - verbose: verbosity level
  - Return
    - None if window_size is set to greater than 0.
    - training_data: Non-sliding windowed and activity segmented sensor data.
    - training_label: Ground truth of each segmented activity.
    - training_subject: Name of subject of each segmented activity.
    - test_data: Non-sliding windowed and activity segmented sensor data. 
    - test_label: Ground truth of each segmented activity. 
    - test_subject: Name of subject of each segmented activity. 

#### 3.2.1.1 Generating TensorFlow-Ready dataset example
```python
from moclaphar.dataset.dataset import make_training_data

make_training_data(data_root="/annotated/data/root/dir/",
                   save_root="/dir/to/store/hdf5/files",
                   window_size=300, stride=90, chunk_size=100)
```

### 3.2.2 moclaphar.dataset.hdf5generator.py
* HDF5Generator(class)
  - TensorFlow-compatible data generator class.
  - Initialization: HDF5Generator(path, prefix, verbose=1)
    - path: .hdf5 file path that generated from make_training_data
    - prefix: can only be set to 'training' or 'test'.
    - verbose: verbosity level
  - Members
    - self.path: .hdf5 file path that generated from make_training_data
    - self.prefix: can only be set to 'training' or 'test'.
    - self.data: hdf5 data
    - self.n_data: Number of data contains in self.data
    - self.class_histogram: Histogram information of classes.
    - self.n_class: Total number of classes in dataset
    - self.class_weight: Weights calculated from class_histogram. Greater weights on smaller samples.
    
#### 3.2.2.1 HDF5Generator usage
```python
from moclaphar.dataset.hdf5generator import HDF5Generator

data_generator = HDF5Generator("/dataset/path", "training")

data_generator.data['training_data']
data_generator.data['training_label']
```

### 3.2.3 moclaphar.dataset.io_helper.py
* append_h5py_data(data, fname, db_key, dtype='float32')
  - Append data to hdf5 file. Since writing large-scale data requires big memory consumption, appending h5py can generate large hdf5 file from PC with low memory.
  - Arguments
    - data: data to append.
    - fname: file path
    - db_key: hdf5 requires key to find data. Using same key will append the data at the last data point.
    - dtype: data type
  
* save_windowed_data_hdf5(data, label, s_idx=-1, e_idx=-1, window_size=300, stride=1, save_root='../data/', prefix="training", verbose=1)
  - Generates sliding windowed data and saves as hdf5 file format.
  - Arguments
    - data: data to be saved.
    - label: ground truth label to be saved.
    - s_idx: starting index. 
    - e_idx: end index. This function will only process data[s_idx:e_idx] and label[s_idx:e_idx]
    - window_size: sliding window size
    - stride: sliding stride size
    - save_root: save data root
    - prefix: prefix of db_key in hdf5 file.
    - verbose: verbosity level
    
* save_windowed_dataset_hdf5(training_data, training_label, test_data, test_label, window_size=300, stride=1, save_root='../data/', chunk_size=100)
  - Generate sliding windows from training and test data and saves as hdf5 file format.
  - Arguments
    - training_data: training data to be saved
    - training_label: ground truth training label to be saved 
    - test_data: test data to be saved  
    - test_label: ground truth test label to be saved
    - window_size: sliding window size
    - stride: sliding stride size
    - save_root: save data root
    - chunk_size: size of processing data at once. Recommended to have lower value from PC with low memory
    
### 3.2.4 moclaphar.dataset.preprocess.py
* normalize_data(data, norm_max=1)
  - Normalize sensor data by each axis's min/max value
  - Arguments:
    - data: data to be normalized
    - norm_max: maximum value after normalized. Range of value will be set to 0~norm_max
* reshape_data
  - Concatenates each sensor axis to one axis. Ex) (n, w, 5, 6) -> (n, w, 30)
    - data: data to be reshaped.
    - rotate: True swaps axes of 1 and 2. 
  - Return
    - data: reshaped data 
    
* generate_sliding_window_data(data, label, window_size=300, stride=1)
  - Generates sliding window data and corresponding label form data and label
  - Arguments
    - data: data to apply sliding window
    - label:  ground truth label data
    - window_size: sliding window size
    - stride: sliding stride size

## 3.3 moclaphar.utils
### 3.3.1 moclaphar.utils.data_loader.py
* read_mat_file
  - Reading annotated .mat file and converts into python-ready format. Annotation .mat file has 2 versions. 
  One generated from MATLAB annotation script and the other one generated from python annotation script.
  Both versions follows same format but there are slight differences. 
  This function handles both types of .mat files.
  
  - Arguments
    - path: mat file path
  - Returns
    - csv_data(dict): Contains full length of sensor data without annotation
      - 'file_name': original .mat file name
      - 'file_path': original .mat file path
      - 'original_data': unprocessed raw data. None if .mat file is generated from python annotator script.
      - 'duration': recording duration in ms 
      - 'sensor_data': sensor data that NaN values are eliminated
      - 'x': timestamp data
    - vid_data(dict): Contains video file name and path
      - 'vid_name': video file name
      - 'vid_path': original video path in annotation
    - segment_data(dict): Contains each segmented activity data from annotation
      - 'video_sync_time'(float): Synchronization time between sensor and video. Ex) -0.5 represents sensor recording was started 0.5 second later than video recording.
      - 'segment_x'(list): Timestamp x vector of each segmentation.
      - 'segment_sensor_data'(list): Each segmented activity sensor data from annotation. 
      - 'segment_label'(list): label id of each segmented activity.
      - 'segment_name'(list): label name of each segmented activity.
* read_csv_file
  - Read original unannotated csv file. One csv file contains 5 sensor data from each body part. This function synchronizes times that each sensor data arrived. In other words, all 5 sensor data will have identical timestamp and sampling rate.   
  - Arguments
    - path: csv file path
    - video_duration(float or None): If this is given, the timestamp of sensor data from csv is fixed to have equal time gaps.
  - Return
    - sensor_data: Sensor data of (n, 11) shape. (n: data size) (11: "SensorIndex", "Timestamp", "accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ", "oriX", "oriY", "oriZ")
    - duration: Duration of sensor data recording in seconds

### 3.3.2 moclaphar.utils.data_plotter
* draw_segmentation(csv_data, segment_data, sensor_idx=range(0,3), figure_size=(20, 10))
  - Draws full length of sensor data and marks each activity location and name of the activity.
  - Arguments
    - csv_data: full length of original data
    - segment_data: segmented activity data from annotation
    - sensor_idx: sensor axis to be plotted. (range(0, 3): accelerometer, range(3, 6): gyroscope, range(6, 9): orientation)
    - figure_size: figure size of matplotlib.pyplot canvas

### 3.3.3 moclaphar.utils.video_segmenter
* generate_segmented_video(vid_info, segment_data, video_root, save_root=None, verbose=0)
  - Generates video files of each segmented activity from annotation
  - Arguments
    - vid_info: video information from read_mat_file function
    - segment_data: each segmented activity sensor data from annotation
    - video_root: original video root path
    - save_root: root path that segmented videos to be saved
    - verbose: verbosity level 

