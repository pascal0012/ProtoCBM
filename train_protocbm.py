import argparse
from argparse import Namespace

import yaml

from models.models import ModelXtoCtoY, ModelXtoC, ModelXtoY


def train(model: nn.Module, args: Namespace) -> float:
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        default="configs/debug.yaml",
        help="Path to config file (YAML)",
    )
    cli_args = parser.parse_args()

    # Load the config yaml
    with open(cli_args.config) as f:
        args = yaml.safe_load(f)

    # Add run name, keep as namespace to be able to access like args.param
    args = Namespace(
        **args, run_name=cli_args.run_name, config_path=cli_args.config
    )

    if args.mode == "XCY":
        model = ModelXtoCtoY(args)
    elif args.mode == "XY":
        model = ModelXtoY(args)
    elif args.mode == "XC":
        model = ModelXtoC(args)

    train(model, args)