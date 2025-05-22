#!/bin/bash
#SBATCH --job-name=calc_attribution_completion
#SBATCH --output="output/%o_calc_attribution_completion_%j.out"
#SBATCH --error="error/err_calc_attribution_completion_%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --time=1:00:00

# module purge
# module load WebProxy
# module load CUDA/12.4.0
# module load GCC/11.3.0
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

PROJECT_DIR="/scratch/user/ajayjagan2511/IRCAN"
cd $PROJECT_DIR
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"


echo "PYTHONPATH: $PYTHONPATH"

COMPUTE_NODE=$(hostname -s)

source ~/.bashrc
conda activate IRCAN


echo "Job started at $(date)"
echo "Running on ${COMPUTE_NODE}.grace2.hprc.tamu.edu"
nvidia-smi

python src/1_calculate_attribution_completion.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --data_path data/MemoTrap/memo-trap_classification.jsonl \
    --model_name llama2-7b \
    --dataset_name Memo \
    --output_dir results/attribution/ \
    --gpu_id 0 \
    --max_seq_length 128 \
    --batch_size 20