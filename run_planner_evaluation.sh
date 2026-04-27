#!/bin/bash

model="gpt-4.1"
modes=("plan" "code-plan")
strategies=("zeroshot")

for mode in "${modes[@]}"; do
    for strategy in "${strategies[@]}"; do
        python3 run_evaluation.py --model $model --mode $mode --provider openai --dataset morehopqa --fewshot-dataset morehopqa --output_file planner --strategy $strategy --max-samples 100
    done
done
