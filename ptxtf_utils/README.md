# PT to/from TF utils
ptxtf_utils is a library of utility tools to convert fault lists from PyTorch to TensorFlow and viceversa.
Each tool is runnable as a standalone Python script. Do note that all input and output files are formatted according to the content in [dnn-benchmarks](https://gitlab.pmcs2i.ec-lyon.fr/spappala/dnn-benchmarks).

## Requirements
ptxtf_net.py runs with any version of PyTorch and any version of TensorFlow2. Run `pip install -r requirements.txt` to ensure you have all the requirements installed. 
ptxtf_fault.py does not have any requirements.

## Modules
- `ptxtf_fault.py`: converts a fault list from PyTorch to TensorFlow and vice-versa. It receives as inputs a fault list and a file containing the match between layers in the two corresponding PyTorch and TensorFlow models; this matching file is the one outputted by `ptxtf_net.py`. Run `python ptxtf_fault.py --help` for more information.
Expected input format for the fault list:

| Injection | Layer  |   TensorIndex   | Bit |
|:---------:|:------:|:---------------:|:---:|
|         0 | conv2d |  "(2, 1, 0, 7)" |  15 |
|         1 | conv2d | "(2, 0, 0, 14)" |   5 |

This script also supports as input any `csv` file containing an arbitrary number of columns after the `Bit` one; in the conversion, these additional columns will be left unchanged. As a consequence, this script can also be used to convert a report produced by a fault injection campaign.

> [!NOTE]
> This script only converts the layer name and permutes TensorIndex; it does not perform any sanity checks on the location specified by the fault.

- `ptxtf_net.py`: outputs a matching of the layers of corresponding networks. For the PyTorch model, a Python file containing its definition is required, whereas for Tensorflow a `.keras` file is needed.  Run `python ptxtf_net.py --help` for more information.
Output format:
```
PT,TF
pt_layer0,tf_layer0
pt_layer1,tf_layer1
...
pt_layerN, tf_layerN
```
> [!WARNING]
> Please be sure of the safety/correctness of the PyTorch script since it will be entirely executed.

> [!NOTE]
> At the moment, only the layers considered in the models in [dnn-benchmarks](https://gitlab.pmcs2i.ec-lyon.fr/spappala/dnn-benchmarks) are supported:
> - PyTorch: nn.Conv2d, nn.Linear 
> - TensorFlow: keras.layers.Conv2D, keras.layers.Dense

- `fault_writer.py`: generates a fault list targeting either a PyTorch or a TensorFlow model. Run `python fault_writer.py --help` for more information.
The output format is the following:

| Injection | Layer  |   TensorIndex   | Bit |
|:---------:|:------:|:---------------:|:---:|
|         0 | conv2d |  "(2, 1, 0, 7)" |  15 |
|         1 | conv2d | "(2, 0, 0, 14)" |   5 |

> [!WARNING]
> Please be sure of the safety/correctness of the PyTorch script since it will be entirely executed.
