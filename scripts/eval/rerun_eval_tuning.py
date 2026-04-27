"""Re-run eval.py on tuning runs that failed due to wrong data_dir."""
import os
import sys
import argparse

from eval import eval as run_evaluation
from argparse import Namespace
from utils_protocbm.train_utils import gather_args


CY_CHECKPOINT = "weights/protoCBM-models/independent/cy_sigmoid.pth"

EXPERIMENTS = [
    ("outputs/protocbm_independent_decorrelation_reg", 24),  # run_0 .. run_23
    ("outputs/protocbm_independent_loss_weights", 27),       # run_0 .. run_26
    ("outputs/protocbm_independent_full_loss_sweep", 24),    # run_0 .. run_23
]


def build_eval_args(experiment_dir: str, run_id: int) -> Namespace:
    run_dir = os.path.join(experiment_dir, f"run_{run_id}")
    model_path = os.path.join(run_dir, "best_model_1.pth")

    if not os.path.exists(model_path):
        return None

    args = Namespace(
        mode="independent",
        model_name="protocbm",
        concept_mapper="protomod",
        concept_activation="none",
        n_attributes=112,
        saliency_method="attention",
        IoU_threshold=0.5,
        plot_curve=False,
        use_argmax=True,
        vis_every_n=0,
        dataset="cub",
        seed=1,
        batch_size=64,
        use_aux=False,
        backbone="inception",
        backbone_pretrained=True,
        backbone_freeze=False,
        expand_dim=0,
        proto_n_vectors=1,
        use_sigmoid_logits=True,
        data_dir="data/CUB_200_2011",
        split_dir="data/CUB_processed/class_attr_data_10/test.pkl",
        log_dir=run_dir,
        xc_checkpoint=model_path,
        cy_checkpoint=CY_CHECKPOINT,
    )

    out_folder_path = os.path.join(run_dir, f"{args.dataset}_visualization_{args.saliency_method}")
    os.makedirs(out_folder_path, exist_ok=True)
    args.out_dir_part_seg = out_folder_path

    return args


def get_all_runs():
    """Return flat list of (experiment_dir, run_id) tuples."""
    runs = []
    for exp_dir, n_runs in EXPERIMENTS:
        for run_id in range(n_runs):
            runs.append((exp_dir, run_id))
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True, help="1-based SLURM array task ID")
    parser.add_argument("--n-tasks", type=int, default=6, help="Total number of array tasks")
    cli_args = parser.parse_args()

    all_runs = get_all_runs()
    # Distribute runs round-robin across tasks
    my_runs = [r for i, r in enumerate(all_runs) if (i % cli_args.n_tasks) == (cli_args.task_id - 1)]

    print(f"Task {cli_args.task_id}/{cli_args.n_tasks}: processing {len(my_runs)} runs")

    for exp_dir, run_id in my_runs:
        print(f"\n=== Evaluating {exp_dir}/run_{run_id} ===")
        args = build_eval_args(exp_dir, run_id)
        if args is None:
            print(f"  Skipping: no model found")
            continue

        eval_txt = os.path.join(args.out_dir_part_seg, "eval.txt")
        # Clear old (broken) eval.txt
        if os.path.exists(eval_txt):
            os.remove(eval_txt)

        original_stdout = sys.stdout
        sys.stdout = open(eval_txt, "w")
        try:
            run_evaluation(args)
        except Exception as e:
            sys.stdout.close()
            sys.stdout = original_stdout
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            continue
        finally:
            if sys.stdout != original_stdout:
                sys.stdout.close()
                sys.stdout = original_stdout

        print(f"  Done -> {eval_txt}")


if __name__ == "__main__":
    main()
