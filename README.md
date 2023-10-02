## Deep Cross-domain Few-shot Learning for Hyperspectral Image Classification
This is a code demo for the paper "Deep Cross-domain Few-shot Learning for Hyperspectral Image Classification"

Some of our code references the projects
* [Learning to Compare: Relation Network for Few-Shot Learning](https://github.com/floodsung/LearningToCompare_FSL.git)


## Requirements
CUDA = 10.0

Python = 3.7 

Pytorch = 1.5 

sklearn = 0.23.2

numpy = 1.19.2

## dataset
1. target domain data set:

You can download the hyperspectral datasets in mat format at: http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes, and move the files to `./datasets` folder.

2. source domain data set:

The source domain  hyperspectral datasets (Chikusei) in mat format is available in:https://pan.baidu.com/s/1Svt-8HC_FY3lQ1opO88X1A?pwd=6j42 
 
You can download the preprocessed source domain data set (Chikusei_imdb_128.pickle) directly in pickle format, which is available in "https://pan.baidu.com/s/1cbVzKSBxcWdOH5xGzwIlgA?pwd=5xk7" , and move the files to `./datasets` folder.

An example dataset folder has the following structure:
```
datasets
├── Chikusei_imdb_128.pickle
├── IP
│   ├── indian_pines_corrected.mat
│   ├── indian_pines_gt.mat
├── salinas
│   ├── salinas_corrected.mat
│   └── salinas_gt.mat
├── pavia
│   ├── pavia.mat
│   └── pavia_gt.mat
└── paviaU
    ├── paviaU_gt.mat
    └── paviaU.mat
```

## Usage:
Take DCFSL method on the UP dataset as an example: 
1. Download the required data set and move to folder`./datasets`.
2. If you down the source domain data set (Chikusei) in mat format,you need to run the script `Chikusei_imdb_128.py` to generate preprocessed source domain data. 
3. Taking 5 labeled samples per class as an example, run `DAFSC-UP.py --test_lsample_num_per_class 5 --tar_input_dim 103`. 
 * `--test_lsample_num_per_class` denotes the number of labeled samples per class for the target domain data set.
 * `--tar_input_dim` denotes the number of bands for the target domain data set.
