#!/bin/bash
# Submit eval.slurm for all 3 seeds of KL-Distance Oracle models
# Then submit eval_sub.slurm for each seed after its eval finishes

EVAL_JOB_IDS=""

for SEED in seed-1 seed-2 seed-3; do
    CONFIG="configs/protocbm/eval/oracle_kl_distance/${SEED}.yaml"

    # Submit eval job
    EVAL_JOB_ID=$(sbatch \
        --job-name="eval_oracle_kl_${SEED}" \
        --export=ALL,CONFIG="${CONFIG}" \
        eval.slurm | awk '{print $4}')

    echo "Submitted eval for ${SEED}: job ${EVAL_JOB_ID}"

    # Submit eval_sub job dependent on eval job
    SUB_JOB_ID=$(sbatch \
        --dependency=afterok:${EVAL_JOB_ID} \
        --job-name="eval_sub_oracle_kl_${SEED}" \
        --export=ALL,CONFIG="${CONFIG}" \
        eval_sub.slurm | awk '{print $4}')

    echo "Submitted eval_sub for ${SEED}: job ${SUB_JOB_ID} (depends on ${EVAL_JOB_ID})"
done
