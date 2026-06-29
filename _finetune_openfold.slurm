#!/bin/bash
#SBATCH --job-name=soloseq_finetune
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --output=data/pdb_recent/soloseq_finetuning_outputs/slurm-%j.out
#SBATCH --error=data/pdb_recent/soloseq_finetuning_outputs/slurm-%j.err
#SBATCH --mail-user=luxwj0628@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Submit with:  sbatch _finetune_openfold.slurm
# Requests both RTX 6000 Ada GPUs on `tornade` (matches the python --gpus 2 flag).

# Note: no `set -u` — conda's activate/deactivate scripts reference unbound
# variables (e.g. CONDA_BACKUP_CXX) and would abort the job under nounset.
set -eo pipefail

# Run from the directory the job was submitted from (the project root).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Activate the openfold conda environment.
source /home/j-wang/miniconda3/etc/profile.d/conda.sh
conda activate openfold_env

echo "Job ${SLURM_JOB_ID} on $(hostname) | GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Runs the active train_openfold.py command, which writes to a fresh dated
# output dir and loads the original weights read-only.
bash _finetune_openfold.sh
