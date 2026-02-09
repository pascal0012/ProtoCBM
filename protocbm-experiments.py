from argparse import Namespace

from train_protocbm import train
from utils_protocbm.train_utils import gather_args, model_by_mode

def main(args: Namespace):
    run_name = args.get("experiment_name", None)
    assert run_name is not None, "experiment_name must be provided in args!"

    for config in args.experiments:
        print(f"Running config: {config['name']}")

        # Deepcopy args to avoid overwriting
        args = Namespace(**vars(args))

        args.log_dir = f"outputs/{run_name}/{config['name']}"
        args.write_console = False

        # Copy over config values
        [setattr(args, key, value) for key, value in config.items() if key != "name"]

        model = model_by_mode(args)

        metric = train(model, args)
        print(f"Final metric for config {config['name']}: {metric}")


if __name__ == "__main__":
    args = gather_args()

    # Load base config if provided
    base_config = args.get("base_config", None)
    if base_config is not None:
        print(f"Loading base config from {base_config}")
        args.update(gather_args(config_file=base_config))

    main(args)
