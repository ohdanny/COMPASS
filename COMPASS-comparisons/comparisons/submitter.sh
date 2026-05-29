#!/bin/bash --login
#SBATCH --job-name=compass-sim
#SBATCH --array=0-647
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/slurm_%x_%A_%a.out
#SBATCH --error=slurm_logs/slurm_%x_%A_%a.err


METHOD="${1:?usage: sbatch submit.sh <compass|spvc|splisosm>}"

mkdir -p slurm_logs

# adjust path if your conda init lives elsewhere
module purge
source ~/.bashrc
conda activate spvapa
module load R/4.5.1-gfbf-2025a
export PATH="/mnt/home/hegazyab/miniconda3/envs/spvapa/bin:$PATH"
# the above makes sure the right python is used

# keep JAX / BLAS / OpenMP honest with what SLURM gave us
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"


CONFIG_ID=$((SLURM_ARRAY_TASK_ID / 8))
TASK_ID=$((SLURM_ARRAY_TASK_ID % 8))


set -euo pipefail


echo "[$(date)] start task=$SLURM_ARRAY_TASK_ID method=$METHOD host=$(hostname) cpus=$SLURM_CPUS_PER_TASK"
python main.py --task-id "$TASK_ID" --method "$METHOD" --config-id "$CONFIG_ID" --output-dir "res/compass_sweep" --log-dir "logs/compass_sweep" --clear-existing
echo "[$(date)] done  task=$SLURM_ARRAY_TASK_ID method=$METHOD"
