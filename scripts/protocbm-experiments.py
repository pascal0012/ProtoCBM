from argparse import Namespace

from train_protocbm import train
from utils_protocbm.train_utils import gather_args, model_by_mode

def main(args: Namespace):
    run_name = args.experiment_name

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

    main(args)
