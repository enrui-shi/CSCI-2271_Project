# CSCI-2271_Project
An pytorch reimplemention of the paper http://vladlen.info/papers/learning-to-see-in-the-dark.pdf pytorch
## requirement
python 3.8
pytorch 1.7
CUDA 11 (for gpu)
glob
rawpy
## data 
[Sony](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)
## traing
Download the Sony data and put it into the data folder, then run the train.py.  Need about 40G ram to load data.
## testing
Run the test.py. You can specify the model to use by changing the model_PATH. Out put will be save to result folder
# Reference

Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.