from argparse import Namespace
from models.models import ModelXtoCtoY
import optuna

from train_protocbm import train
from utils_protocbm.train_utils import gather_args

def objective(args: Namespace, trial: optuna.Trial):
    # Deepcopy args to avoid overwriting
    args = Namespace(**vars(args))

    # Avoid console issues
    args.write_console = False
    
    # Update log dir to avoid overwriting
    args.log_dir = f"outputs/hyperparameter/run_{trial.number}"
    
    # General arguments to optimize
    args.lr = trial.suggest_categorical("lr", [0.01, 0.001, 0.0001])  # type: ignore
    # args.scheduler_step = trial.suggest_categorical( # type: ignore
    #     "scheduler_step", [10, 15, 20, 1000]
    # )
    # args.weight_decay = trial.suggest_categorical("weight_decay", [0.0004, 0.00004])  # type: ignore

    # ProtoMod specific properties
    args.proto_n_vectors = trial.suggest_int("proto_n_vectors", 1, 10)  # type: ignore
    args.loss_decorrelation_per_group = trial.suggest_categorical("loss_decorrelation_per_group", [True, False])  # type: ignore

    # Weighting of loss terms
    args.loss_weight_attribute_reg = trial.suggest_float( # type: ignore
        "loss_weight_attribute_reg", 1e-4, 5, log=True
    )  
    args.loss_weight_map_compactness = trial.suggest_float("loss_weight_map_compactness", 1e-4, 2, log=True) # type: ignore
    args.loss_weight_attribute_decorrelation = trial.suggest_float("loss_weight_attribute_decorrelation", 1e-6, 2) # type: ignore

    model = ModelXtoCtoY(args)

    best_val_metric, val_class_acc_meter, val_attr_acc_meter = train(model, args)
    print(f"Best val metric: {best_val_metric.item():.4f}")
    print(f"Final class accuracy: {val_class_acc_meter.item():.4f}")
    print(f"Final attribute accuracy: {val_attr_acc_meter.item():.4f}")
    return best_val_metric


if __name__ == "__main__":
    args = gather_args()
    args.save_model = False
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(args, trial), n_trials=50)

    print("Best:", study.best_trial.params)
