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
    args.weight_decay = trial.suggest_categorical("weight_decay", [0.0004, 0.00004])  # type: ignore

    # ProtoMod specific properties
    args.proto_n_vectors = trial.suggest_int("proto_n_vectors", 1, 10)  # type: ignore
    args.loss_decorrelation_per_group = trial.suggest_categorical("loss_decorrelation_per_group", [True, False])  # type: ignore

    # Weighting of loss terms
    args.loss_weight_attribute_reg = trial.suggest_float( # type: ignore
        "loss_weight_attribute_reg", 1e-6, 1, log=True
    )  
    args.loss_weight_map_compactness = trial.suggest_float("loss_weight_map_compactness", 1e-9, 1e-1, log=True) # type: ignore
    args.loss_weight_attribute_decorrelation = trial.suggest_float("loss_weight_attribute_decorrelation", 1e-6, 1e-1) # type: ignore

    model = ModelXtoCtoY(args)

    metric, part_seg_iou = train(model, args)

    if args.use_localization.metric:
        return part_seg_iou
    return metric


if __name__ == "__main__":
    args = gather_args()
    args.use_localization_metric = False
    study = optuna.create_study(direction="maximize" if args.use_localization_metric else "minimize")
    study.optimize(lambda trial: objective(args, trial), n_trials=50)

    print("Best:", study.best_trial.params)
