# Model converter
Torch to Keras model converter based on [nobuco](https://github.com/AlexanderLutsenko/nobuco).


## Setup
1. Ensure you have Python 3.9 installed in your working environment
2. Create a virtual environment
```
python -m venv venv_name
source venv_name/bin/activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
## Usage
Run as a python module:
```
python -m model_conversion [ARGS, ...]
```
To display the complete user guide, type
```
python -m model_conversion --help
```

## Supported models
- CIFAR10
    - DenseNet
    - GoogLeNet
    - Inception
    - MobileNetV2
    - ResNet
    - Vgg
- CIFAR100
    - DenseNet 
    - GoogLeNet
    - ResNet
- GTSRB
    - DenseNet`
    - Resnet
    - Vgg

