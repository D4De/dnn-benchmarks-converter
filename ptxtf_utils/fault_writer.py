import os
import sys
import importlib
import pathlib
import numpy as np
import csv
import argparse

from tensorflow import keras
from torch import nn

from typing import Sequence


def generate_coords(
    limits: Sequence[int],
    n: int,
    generator: np.random.Generator,
    bit_dim: int = 32,
) -> np.ndarray:
    """
    Generates n random coordinates inside the interval [0, limits[i]] for the i-th column
    """
    coords = np.zeros(
        (len(limits) + 1, n), dtype=np.uint32
    )  # array with n rows and a column for each dimension
    limits = list(limits)
    limits.append(bit_dim)
    for i in range(coords.shape[0]):
        coords[i] = generator.integers(limits[i], size=n, dtype=np.uint32)

    coords = coords.T
    return coords


REPORT_HEADER = (
    "inj_id",
    "layer_weigths",
    "bit_pos",
    "n_injections",
    "top_1_correct",
    "top_5_correct",
    "top_1_robust",
    "top_5_robust",
    "masked",
    "non_critical",
    "critical",
)


FAULT_HEADER = ["Injection", "Layer", "TensorIndex", "Bit"]


def write_injection(
    layers: Sequence[tuple[str, tuple[int, ...]]],
    output: os.PathLike,
    injections_per_layer: int,
):
    path = pathlib.Path(output)
    if not os.path.exists(path.parents[0]):
        os.makedirs(path.parents[0])

    with open(path, "w") as target:
        writer = csv.writer(target)
        writer.writerow(REPORT_HEADER)
        index = 0
        generator = np.random.default_rng()
        for name, dims in layers:
            injections = generate_coords(dims, injections_per_layer, generator)
            for inj in range(injections_per_layer):
                weight_coord = f"({','.join((str(i) for i in injections[inj, :-1]))})"  # (coord1, ..., coordn)
                writer.writerow((index, name, weight_coord, injections[inj, -1]))
                index += 1


tf_layers = (keras.layers.Conv2D, keras.layers.Dense)
pt_layers = (nn.Conv2d, nn.Linear)


def get_tf_layers(path: os.PathLike):
    model = keras.models.load_model(path)
    layer_coords = [
        (layer.name, layer.get_weights()[0].shape)
        for layer in model._flatten_layers(include_self=False, recursive=True)
        if isinstance(layer, tf_layers)
    ]

    return layer_coords


def get_pt(path: str, name: str):
    """
    Executes at runtime a python module saved in path containing a Torch network of given name. Returns such network. Ensure the execution module has a trusted origin.
    """
    spec = importlib.util.spec_from_file_location("pt_network", path)
    if spec is None:
        raise FileNotFoundError(f"Cannot find file {path}")

    pt_network = importlib.util.module_from_spec(spec)
    sys.modules["pt_network"] = pt_network

    if spec.loader is None:
        raise ValueError("Cannot load module")

    spec.loader.exec_module(pt_network)
    return getattr(pt_network, name)


def get_pt_layers(network: nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    """
    Returns a list containing the association of <layer_name>, <layer_dims> of a given Torch network. See pt_layers for supported layers
    Args:
        network: a torch.nn.Module object
    Returns:
        a list of 2D tuples. Each tuple contains the corresponding layer's name and weight dimensions.
    """
    return [
        (name, module.weight.shape)
        for name, module in network().named_modules()
        if isinstance(module, pt_layers)
    ]


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="fault_writer",
        description="Writes a fault writer targeting either a PT or a TF network. See the help message for 'pt' or 'tf' for more information",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="framework")

    tf_parser = subparsers.add_parser("tf")
    tf_parser.add_argument("filepath", help="path to .keras file")

    pt_parser = subparsers.add_parser("pt")
    pt_parser.add_argument(
        "source",
        nargs=2,
        help="path to a .py file where the network is defined and the name of the network",
    )
    parser.add_argument("--output", "-o", default="./out.csv", help="output filepath")
    parser.add_argument(
        "--ninj", "-n", type=int, default=32, help="number of injections for each layer"
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="random seed for determinism"
    )
    return parser.parse_args(args)


def main(args):
    if args.framework == "tf":
        layers = get_tf_layers(args.filepath)
    elif args.framework == "pt":
        pt_net = get_pt(*args.source)
        layers = get_pt_layers(pt_net)
    else:
        raise ValueError("framework can only be tf or pt")

    if args.seed:
        np.random.seed(args.seed)

    write_injection(layers, args.output, args.ninj)


if __name__ == "__main__":
    main(parse_args())
