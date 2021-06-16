##Class Incremental Learning For Video Action Recognition
###Introduction
This paper presents a CIL method for video action classification. 
This method uses CNN to extract features for each frame, and combines 
the advantages of GWR that can automatically grow or shrink to model the
feature manifold for each action class. The Knowledge Consolidation (KC)
method is introduced to alleviate forgetting by separating the feature
manifolds of old class and new class.

###Code Environment
* ubuntu 16.04
* python 3.6
* numpy 1.19
* pytorch 1.3

###Run
* Download the project and action datasets.

  The download url is [dataset](www.). Extract password is
* Activate your environment
* Modify the absolute path in the project

```
cd ./CIL-for-video-action-recognition
```
```
python train.py
```





