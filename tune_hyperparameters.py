from argparse import Namespace

import optuna

from train_protocbm import model_by_mode, train
from utils_protocbm.train_utils import gather_args


def objective(args: Namespace, trial: optuna.Trial):
    # Deepcopy args to avoid overwriting
    args = Namespace(**vars(args))

    # Avoid console issues
    args.write_console = False

    # Update log dir to avoid overwriting
    args.log_dir = f"outputs/{args.experiment_name}/run_{trial.number}"

    for param_name, param_config in args.optimize.items():
        if param_config["distribution"] == "loguniform":
            value = trial.suggest_float(
                param_name, param_config["low"], param_config["high"]
            )
        elif param_config["distribution"] == "uniform":
            value = trial.suggest_uniform(
                param_name, param_config["low"], param_config["high"]
            )
        elif param_config["distribution"] == "categorical":
            value = trial.suggest_categorical(param_name, param_config["values"])
        else:
            raise ValueError(
                f"Unsupported tuning distribution type: {param_config['distribution']} for parameter {param_name}!"
            )

        setattr(args, param_name, value)

    model = model_by_mode(args)

    best_val_metric, val_class_acc_meter, val_attr_acc_meter = train(model, args)

    print(f"----- Trial {trial.number} finished -----")
    print(f"Best val metric for run: {best_val_metric.item():.4f}")
    print(f"Final class accuracy: {val_class_acc_meter.item():.4f}")
    print(f"Final attribute accuracy: {val_attr_acc_meter.item():.4f}")
    print("-----------------------------------------")

    return best_val_metric


if __name__ == "__main__":
    args = gather_args()

    print(f"Tuning hyperparameters for experiment '{args.experiment_name}'.")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(args, trial), n_trials=args.num_trials)

    best_run = study.best_trial
    print(
        f"Best config was number {best_run.number} with value {best_run.value:.4f} and params: {best_run.params}"
    )
