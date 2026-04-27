#!/bin/bash

models=("gpt-4.1-plan" "gpt-4.1-code-plan")
strategies=("zeroshot")

for model in "${models[@]}"; do
    for strategy in "${strategies[@]}"; do
        python3 run_evaluation.py --model $model --dataset morehopqa --fewshot-dataset morehopqa --output_file planner --strategy $strategy --max-samples 100
    done
done
