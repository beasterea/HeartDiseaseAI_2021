### Heart Disease AI Competition using SarUNet
This repository contains the code for the image segmentation for the 2021 Heart Disease AI Competition implemented in Pytorch

### Requirements
- PyTorch 1.x

### Installation
1. Create a conda (GPU) enviornment
```
conda create -n=<enviornment_name> python=3.8 anaconda
conda activate <enviornment_name>
```

2. Install PyTorch
```
conda install pytorch torchvision 

3. Install requirements
```
pip install -r requirements.txt
```

4. Enable the in the bash command
```
python3 test.py 
```
**The Dataset Directory for the Testing Should Be**
- The input image should be in the '.png' format
= The result would be stored in a new folder as image form