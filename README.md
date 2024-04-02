## Pytorch Implementation of "Contrast enhancement: Cross-modal learning approach for medical images"

## Requirements

- Python 3.6
- CUDA 10.0
- To install requirements：
  `pip install -r requirements.txt`

## Prerequisites

### Data

Input for training model, that is the low-contrast images and label, that is high-contrast images are placed in the folder 'images_LR'.
This data can be downloaded from the following link:

[Data](#https://drive.google.com/file/d/1QU-b8BKOrcgbGCwIoNfgXm_vfVksPMxz/view?usp=drive_link) 

### Folders

1. All hyperparameters are in `libs\constant.py`


All this extra foders are created by calling

```python directory_strcture.py```


2. There are some folders need to be created, to do that just call python directory_structure.py:
   1. `images_LR`：Used to store datasets
      1. `Reference`
      2. `input`
      3. In each of the above two folders, the following three new folders need to be created:
         1. `Testing`
         2. `Training1`
         3. `Training2`
   2. `models`：Used to store all the training generated files：
      1. `gt_images`
      2. `input_images`
      3. `pretrain_checkpoint`
      4. `pretrain_images`
      5. `test_images`
      6. `train_checkpoint`
      7. `train_images`
      8. `train_test_images`
    
   3. `model`: Used to store `log_PreTraining.txt`

## Training

1. 1. `images_LR/input/Training1 ` should contain folders containing images you want to correct.
   2. `images_LR/Reference/Training2 ` should contain folders containing  the images of the type you want to obtain.
   3. `images_LR/input/Testing ` `images_LR/Reference/Testing ` should contain folders containing the images to get the test scores. In the former you should put the 'low-contrast' images and in the latter the corresponding 'high-contrast' images.

2. Run the following line to get all the images to max_size 512 ...change the commented directoies as needed
```python resize.py ```

3. Run for training   

```python CE_Train.py ```

## Evaluation

For now, the evaluation and training are simultaneous. So there is no need to run anything. 
Please cite our paper if you find the work useful:

Naseem, R., Islam, A.J., Cheikh, F.A. and Beghdadi, A., 2022. Contrast enhancement: Cross-modal learning approach for medical images. Electronic Imaging, 34, pp.1-6.


