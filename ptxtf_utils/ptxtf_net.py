import importlib.util
import sys
import argparse
from collections import OrderedDict

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from torch import nn


def get_pt(path: str, name: str):
    """
    Executes at runtime a python module saved in path containing a Torch network of given name. Returns such network.
    The name research is case-insensitive and strips any underscore.
    Ensure the execution module has a trusted origin.
    """
    spec = importlib.util.spec_from_file_location("pt_network", path)
    if spec is None:
        raise FileNotFoundError(f"Cannot find file {path}")

    pt_network = importlib.util.module_from_spec(spec)
    sys.modules["pt_network"] = pt_network

    if spec.loader is None:
        raise ValueError("Cannot load module")

    spec.loader.exec_module(pt_network)
    attrs = dir(pt_network)
    for attr in attrs:
        if attr.lower().replace("_", "") == name.lower().replace("_", ""):
            return getattr(pt_network, attr)
    raise AttributeError(f"module {path} has no matching attribute for {name}")


def get_tf(path: str):
    return keras.models.load_model(path)


def get_pt_modules(pt_network, layers):
    names = []
    modules = {}
    for name, module in pt_network().named_modules():
        if isinstance(module, layers):
            names.append(name)
            modules[name] = module
    return names, modules 


def get_tf_modules(tf_network, layers):
    names = []
    modules = {} 
    for layer in tf_network._flatten_layers(include_self=False, recursive=True):
        if isinstance(layer, layers):
            names.append(layer.name)
            modules[layer.name] = layer

    return names, modules


TF_LAYERS = (keras.layers.Conv2D, keras.layers.Dense)
PT_LAYERS = (nn.Conv2d, nn.Linear)

def natsort(layers):
    od = OrderedDict()
    for layer in layers:
        if "_" in layer:
            prefix, suffix = layer.split("_")
            suffix = int(suffix)
        else:
            prefix = layer
            suffix = -1

        if prefix in od.keys():
            od[prefix].append(suffix)
        else:
            od[prefix] = [suffix]
    
    new_layers = []
    for k in od:
        od[k].sort()
        new_layers.extend((f"{k}{'_'+str(v) if v>=0 else ''}" for v in od[k]))

    return new_layers

def permuter(coords):
    if len(coords) == 2:
        permuted = (coords[1], coords[0])
    elif len(coords) == 4:
        permuted = (coords[2], coords[3], coords[1], coords[0])
    else:
        raise ValueError(
            f"unsupported coordinate format: expected 2D or 4D, got {len(coords)}D"
        )
    return permuted



def validate_match(tfl, ptl, tf_names, pt_names):
    error = False
    for tfn, ptn in zip(tf_names, pt_names):
        tf_layer = tfl[tfn]
        pt_layer = ptl[ptn]

        tf_shape = tf_layer.get_weights()[0].shape 
        pt_shape = permuter(tuple(pt_layer.weight.size()))
        if not all(map(lambda t : t[0]==t[1], zip(tf_shape,pt_shape))):
            print(f"ERROR: shape mismatch TF({tfn}:{tf_shape}), PT({ptn}:{pt_shape})")
            error=True
        else:
            print(f"OK: shape match TF({tfn}:{tf_shape}), PT({ptn}:{pt_shape}")
    
    return error

def main(args):
    pt_network = get_pt(*args.pt_path)
    tf_network = get_tf(args.tf_path)

    pt_names, pt_layers = get_pt_modules(pt_network, PT_LAYERS)
    tf_names, tf_layers = get_tf_modules(tf_network, TF_LAYERS)

    if len(pt_names) != len(tf_names):
        print("WARNING: layers differ")

    # enable this assert to block execution when a mismatch is detected
    ### assert len(pt_names) == len(
    #    tf_names
    # ), f"Cannot match layers: expected the same length, got {len(pt_names)}(PT) vs {len(tf_names)}(TF)"
    
    if args.strategy == "natsort":
        tf_names = natsort(tf_names)
    
    error = validate_match(tf_layers, pt_layers, tf_names, pt_names)
    if error and args.strategy == "fallback":
        print("ERROR: detected shape mismatch, trying natsort ordering")
        tf_names = natsort(tf_names)
        error = validate_match(tf_layers, pt_layers, tf_names, pt_names)

    if error:
        raise ValueError("FATAL: could not fix the mismatch")


    with open(args.output, "w") as f:
        f.write("PT,TF\n")
        f.writelines((f"{ptn},{tfn}\n" for ptn, tfn in zip(pt_names, tf_names)))


def parse_args(args=None):
    argparser = argparse.ArgumentParser(
        prog="ptxtf_net",
        description="Matches PT and TF layers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "pt_path",
        nargs=2,
        help="path to a .py file containing a PT network definition and the name of the model",
    )
    argparser.add_argument(
        "tf_path", help="path to a .keras file containing a keras model"
    )
    argparser.add_argument(
        "--output",
        "-o",
        help="save the output of the matching here",
        default="./out.txt",
    )
    argparser.add_argument(
        "--strategy",
        "-s",
        help="the matching strategy: fallback will try juxtapose and then natsort",
        choices=("juxtapose","natsort", "fallback"),
        default="fallback",
    )
    return argparser.parse_args(args)


if __name__ == "__main__":
    main(parse_args())
