# moclaphar
Motion Classification Human Activity Recognition Helper Package

# Usage
## dataset
### HDF5Generator
```python
from moclaphar.dataset.hdf5generator import HDF5Generator

dataset = HDF5Generator("/dataset/path", "training")

dataset.data['training_data']
dataset.data['training_label']
```


### make_training_data
```python
from moclaphar.dataset.dataset import make_training_data

make_training_data(data_root="/annotated/data/root/dir/",
                   save_root="/dir/to/store/hdf5/files",
                   window_size=300, stride=90, chunk_size=100)
```

## annotator
### run_annotator
```python
from moclaphar.annotator import run_annotator

run_annotator("/video/file/path/", "/data/file/path/csv/or/mat/", vw=1024)
```
or
```python
from moclaphar.annotator import run_annotator

run_annotator("", "", vw=1024)
```
The file selector dialogue will prompt.

# Getting started
## pip install
```
pip install moclaphar
```

## Environments
- Python 3.6

## Dependencies
```
pip install -r requirements.txt
```
