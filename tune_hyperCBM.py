from argparse import Namespace
from models.models import ModelXtoCtoY
import optuna

from train_cbm import train
from utils_protocbm.train_utils import gather_args


def objective(args: Namespace, trial: optuna.Trial):
    # Deepcopy args to avoid overwriting
    args = Namespace(**vars(args))

    # Avoid console issues
    args.write_console = False

    # Update log dir to avoid overwriting
    args.log_dir = f"outputs/hyperparameter/CBM/run_{trial.number}"

    # General arguments to optimize
    args.lr = trial.suggest_categorical("lr", [0.01, 0.005, 0.001, 0.0001])  # type: ignore
    args.weight_decay = trial.suggest_categorical("weight_decay", [0.01, 0.001, 0.0004, 0.00004])  # type: ignore

    # CBM specific properties
    args.optimizer = trial.suggest_categorical("optimizer", ["sgd", "Adam", "AdamW"])  # type: ignore

    # Weighting of loss terms
    args.loss_attr_weight = trial.suggest_float(  # type: ignore
        "loss_attr_weight", 1e-6, 1, log=True
    )
    args.attr_loss_weight = trial.suggest_float(  # type: ignore
        "attr_loss_weight", 1e-6, 1, log=True
    )

    model = ModelXtoCtoY(args)
    metric = train(model, args)
    return metric


if __name__ == "__main__":
    args = gather_args()
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(args, trial), n_trials=50)

    print("Best:", study.best_trial.params)
