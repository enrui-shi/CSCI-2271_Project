# CSCI-2271_Project
An pytorch reimplemention of the paper http://vladlen.info/papers/learning-to-see-in-the-dark.pdf 
## Requirement
python 3.8  
pytorch 1.7  
CUDA 11 (for gpu)  
glob  
rawpy  
RAM (>40G)  (loading traing data)
## Data 
[Sony](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)
## Traing
Download the Sony data and put it into the data folder, then run the
```Shell
train.py
```
By defalut it will run for 8000 epoch and save the model for every 200 epoch
## Testing
```Shell
test.py
```
You can specify the model to use by changing the model_PATH. Out put will be save to result folder
# Reference

Chen Chen, Qifeng Chen, Jia Xu, and Vladlen Koltun, "Learning to See in the Dark", in CVPR, 2018.