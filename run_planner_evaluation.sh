#!/bin/bash

models=("gpt-4o-plan" "gpt-4o-code-plan")
strategies=("zeroshot" "2-shot" "3-shot" "zeroshot-cot" "2-shot-cot" "3-shot-cot")

for model in "${models[@]}"; do
    for strategy in "${strategies[@]}"; do
        python3 run_evaluation.py --model $model --dataset morehopqa --fewshot-dataset morehopqa --output_file planner --strategy $strategy
    done
done
