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
```

3. Install requirements
```
pip install -r requirements.txt
```
```
current working directory
|__test.py
|__model.py
|__crop_sarunet_A2C_bce01.pth
|__sarunet_A4C.pth
|__data_path
    |__test
        |__A2C
            |__result (MAKE A FOLDER TO SAVE THE RESULT MASKS FROM THE NET WITH THE NAME 'result')
        |__A4C
            |__result (MAKE A FOLDER TO SAVE THE RESULT MASKS FROM THE NET WITH THE NAME 'result')
```

4. Enable in the bash command
```
python3 test.py --name [name of the model] --data_path[absolute path of the dataset] --data_type ['validation' / 'test'] --image_type['A2C' / 'A4C']
```
**The Dataset Directory for the Testing Should Be**
- The input image should be in the '.png' format
= The result would be stored in a new folder as image form

```
data_path (absolute path of the dataset)
|__data_type
    |__A2C
    |   |_..png
    |   |_..npy
    |   .
    |   .
    |__A4C
        |_..png
        |_..npy
```