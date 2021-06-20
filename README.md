## Class Incremental Learning For Video Action Classification
### Introduction
This paper presents a CIL method for video action classification. 
This method uses CNN to extract features for each frame, and combines 
the advantages of GWR that can automatically grow or shrink to model the
feature manifold for each action class. The Knowledge Consolidation (KC)
method is introduced to alleviate forgetting by separating the feature
manifolds of old class and new class.

### Code Environment
* ubuntu 16.04
* python 3.6
* numpy 1.19
* pytorch 1.3

###  prepare
* Download the project and action datasets.

  The download url is [dataset](https://pan.baidu.com/s/1qBXWKJUbfPzMWetK_vSEpA). Extract password is fffv
* Activate your environment
* Modify the absolute path in the project

### train and test for each incremental session
```
cd ./CIL-for-video-action-recognition
```
```
python train.py
```

### paper information
Title: CLASS INCREMENTAL LEARNING FOR VIDEO ACTION CLASSIFICATION (Accepted by icip 2021)  
Authors: Jiawei Ma, Xiaoyu Tao, Jianxing Ma, Xiaopeng Hong, Yihong Gong


### Contact Email
If you have questions, please contact the Email:  dixian_1996@163.com




