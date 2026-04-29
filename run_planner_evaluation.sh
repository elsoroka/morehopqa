#!/bin/bash

model="gpt-4o"
modes=("default" "code-plan")
strategies=("zeroshot")

for mode in "${modes[@]}"; do
    for strategy in "${strategies[@]}"; do
        python3 run_evaluation.py --model $model --mode $mode --provider openai --dataset morehopqa --fewshot-dataset morehopqa --output_file $mode --strategy $strategy --max-samples 100
    done
done

python3 run_evaluation.py --model gpt-4.1 --mode default --provider openai --dataset morehopqa --fewshot-dataset morehopqa --output_file default --strategy zeroshot --max-samples 100