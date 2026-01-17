from argparse import Namespace

from train_protocbm import train
from utils_protocbm.train_utils import gather_args, model_by_mode

# Model Type, Attribute Regression Loss, Attribute Decorrelation Loss, Map Compactness Loss
attribute_reg_weight = 0.8886
attribute_decorrelation_weight = 0.0088
map_compactness_weight = 0.01

run_configs = [
    {
        "id": "XCY-0",
        "mode": "XCY",
        "loss_weight_attribute_reg": 0,
        "loss_weight_attribute_decorrelation": 0,
        "loss_weight_map_compactness": 0,
    },
    {
        "id": "XCY-1",
        "mode": "XCY",
        "loss_weight_attribute_reg": attribute_reg_weight,
        "loss_weight_attribute_decorrelation": 0,
        "loss_weight_map_compactness": 0,
        "proto_n_vectors": 1,
    },
    {
        "id": "XCY-2",
        "mode": "XCY",
        "loss_weight_attribute_reg": attribute_reg_weight,
        "loss_weight_attribute_decorrelation": attribute_decorrelation_weight,
        "loss_weight_map_compactness": 0,
        "proto_n_vectors": 1,
    },
    {
        "id": "XCY-3",
        "mode": "XCY",
        "loss_weight_attribute_reg": attribute_reg_weight,
        "loss_weight_attribute_decorrelation": attribute_decorrelation_weight,
        "loss_weight_map_compactness": map_compactness_weight,
        "proto_n_vectors": 1,
    },
    {
        "id": "XC-3",
        "mode": "XC",
        "loss_weight_attribute_reg": attribute_reg_weight,
        "loss_weight_attribute_decorrelation": attribute_decorrelation_weight,
        "loss_weight_map_compactness": map_compactness_weight,
        "proto_n_vectors": 1,
    },
    {
        "id": "CY",
        "mode": "CY",
        "lr": 0.01,
        "scheduler_step": 1000,
        "loss_weight_attribute_reg": 0,
        "loss_weight_attribute_decorrelation": 0,
        "loss_weight_map_compactness": 0,
        "checkpoint": "outputs/midterm/XC-3/best_model_1.pth",
        "proto_n_vectors": 1,
    },
    {
        "id": "XC-3-v4",
        "mode": "XC",
        "loss_weight_attribute_reg": attribute_reg_weight,
        "loss_weight_attribute_decorrelation": attribute_decorrelation_weight,
        "loss_weight_map_compactness": map_compactness_weight,
        "proto_n_vectors": 4,
    },
    {
        "id": "CY-v4",
        "mode": "CY",
        "lr": 0.01,
        "scheduler_step": 1000,
        "loss_weight_attribute_reg": 0,
        "loss_weight_attribute_decorrelation": 0,
        "loss_weight_map_compactness": 0,
        "checkpoint": "outputs/midterm/XC-3-v4/best_model_1.pth",
        "proto_n_vectors": 4,
    },
] + [
    {
        "id": f"XCY-3-v{i}",
        "mode": "XCY",
        "loss_weight_attribute_reg": attribute_reg_weight,
        "loss_weight_attribute_decorrelation": attribute_decorrelation_weight,
        "loss_weight_map_compactness": map_compactness_weight,
        "proto_n_vectors": 1,
    }
    for i in range(2, 11)
]


def main(args: Namespace):
    for config in run_configs:
        print(f"Running config: {config['id']}")

        # Deepcopy args to avoid overwriting
        args = Namespace(**vars(args))

        args.log_dir = f"outputs/midterm/{config['id']}"
        args.write_console = False

        # Copy over config values
        [setattr(args, key, value) for key, value in config.items() if key != "id"]

        model = model_by_mode(args)

        metric = train(model, args)
        print(f"Final metric for config {config['id']}: {metric}")


if __name__ == "__main__":
    args = gather_args()

    main(args)
