import argparse
import csv
from typing import Callable, TypeVar


def create_matcher(file, rtl=True) -> dict[str, str]:
    matcher = {}
    reader = file.readlines()
    next(iter(reader))  # skip header
    for line in reader:
        pt, tf = line.split(",")
        tf = tf.strip()
        if rtl:  # tf->pt
            pt, tf = tf, pt
        matcher[pt] = tf
    return matcher


def translate_fault(
    reader,
    writer,
    layer_matcher: dict[str, str],
    permutation: Callable[[tuple], tuple],
    skip_row: bool = False,
):
    """
    Translates a fault list in csv format, applying bit_permutation to coordinates.
    Args:
        reader: iterable input object
        writer: csv._writer object
        layer_matcher: dict containing matching layers (in->out)
        bit_permutation: a callable that permutes coordinates
        skip_row: skip the second row (gold row in injector reports)
    """
    writer.writerow(next(iter(reader)))  # header
    if skip_row:
        writer.writerow(next(iter(reader)))  # gold row

    for row in reader:
        id, layer, str_coords, bit = row[:4]
        if len(row) > 4:
            remainder = row[4:]
        else:
            remainder = ()
        layer = layer_matcher[layer]
        coords = tuple((int(coord) for coord in str_coords[1:-1].split(",")))
        coords = permutation(coords)
        writer.writerow((id, layer, coords, bit, *remainder))


CoordT = TypeVar("CoordT", tuple[int, int], tuple[int, int, int, int])


def permuter(coords: CoordT) -> CoordT:
    # permute(2, 1, 0)
    if len(coords) == 2:
        permuted = (coords[1], coords[0])
    elif len(coords) == 4:
        permuted = (coords[3], coords[2], coords[0], coords[1])
    else:
        raise ValueError(
            f"unsupported coordinate format: expected 2D or 4D, got {len(coords)}D"
        )
    return permuted


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        prog="ptxtf_fault",
        description="Converts a faultlist between PT and TF. As default, translates from PT to TF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file_path", help="path to a csv fault list")
    parser.add_argument(
        "--pt",
        "-p",
        action="store_true",
        help="if set, translates a tf faultlist to a pt one",
    )
    parser.add_argument(
        "--no-permute", "-n", action="store_true", help="if set, don't permute coords"
    )
    parser.add_argument(
        "--layers", "-l", help="file containing a pt and tf layers matching"
    )
    parser.add_argument(
        "--from-report",
        "-r",
        action="store_true",
        help="pass to skip the gold row in the injector report",
    )
    parser.add_argument("--output", "-o", help="output file path", default="./out.csv")
    return parser.parse_args(args)


def main(args):
    with open(args.layers, "r") as f:
        matcher = create_matcher(f, args.pt)

    if args.no_permute:
        permuter_f = lambda x: x
    else:
        permuter_f = permuter

    with open(args.file_path, "r") as fr:
        with open(args.output, "w") as fw:
            translate_fault(
                csv.reader(fr), csv.writer(fw), matcher, permuter_f, args.from_report
            )


if __name__ == "__main__":
    main(parse_args())
