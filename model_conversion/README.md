# Model converter
Tool for converting a PyTorch model into an equivalent TensorFlow one (based on Keras). The tool is based on [nobuco](https://github.com/AlexanderLutsenko/nobuco); a public backup fork of nobuco is also available at the following link [nobuco-fork](https://github.com/D4De/nobuco).


## Setup
1. Ensure you have Python 3.9 installed in your working environment.
2. Create a virtual environment:
```
python -m venv venv_name
source venv_name/bin/activate
```
3. Install dependencies:
```
pip install -r requirements.txt
```
## Usage
Run as a python module:
```
python -m model_conversion [ARGS, ...]
```
To display the complete user guide, type:
```
python -m model_conversion --help
```

## Tested models
The converter has been currently tested on the models contained in [dnn-benchmarks](https://gitlab.pmcs2i.ec-lyon.fr/spappala/dnn-benchmarks). In particular:
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
    - DenseNet
    - Resnet
    - Vgg
- PascalVOC
    - DeepLabV3
