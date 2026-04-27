#!/bin/sh  
#SBATCH --job-name=deepseek-coder-33b-vllm
#SBATCH -p batch  
#SBATCH --nodes=1
#SBATCH -A marlowe-m000186-pm05
#SBATCH -G 4
#SBATCH -c 8
#SBATCH --mem-per-gpu=64G
#SBATCH --time=2:00:00
#SBATCH --output=vllm_deepseek-coder-33b-instruct_%A.out
#SBATCH --error=vllm_deepseek-coder-33b-instruct_error_%A.out
#SBATCH --export=ALL

module load conda
CONDA_BASE="/marlowe/apps/Mambaforge/24.3.0-0"
eval "$($CONDA_BASE/bin/conda shell.bash hook)"
conda activate /scratch/m000186/esoroka/

vllm serve "deepseek-ai/deepseek-coder-33b-instruct" --port 8000 \
      --tensor-parallel-size 4 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 32768 &

echo "Waiting for vLLM to be ready..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 5
done
echo "vLLM is ready"

# Run the evaluation script
python3 run_evaluation.py --model deepseek-ai/deepseek-coder-33b-instruct --mode code-plan --provider vllm --dataset morehopqa --fewshot-dataset morehopqa --output_file deepseek-coder-33b-instruct_code_plan --strategy zeroshot --max-samples 100
