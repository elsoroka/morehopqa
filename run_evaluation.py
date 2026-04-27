"""Run evaluation on dataset

Format: python3 run_evaluation.py --model ... --mode ... --provider ... --dataset ... --strategy ...
"""
import argparse
import os
from dotenv import load_dotenv
load_dotenv()
from evaluate import evaluate_all, evaluate_baseline
from datasets.abstract_dataset_loader import DatasetLoader
from models.abstract_model import AbstractModel
from models.prompt_generator import PromptGenerator
import sys
from datetime import datetime
from postprocess import postprocess_all, postprocess_all_baseline
import json

class DatasetSlice:
    """Wraps a DatasetLoader and limits the number of items returned."""
    def __init__(self, dataset, max_samples):
        self._items = list(dataset.items())[:max_samples]
        self.length = len(self._items)

    def items(self):
        return iter(self._items)


def main():
    parser = argparse.ArgumentParser(description="Process model and dataset flags.")
    parser.add_argument('--model', type=str, help='Model name to use (e.g. gpt-4.1, Qwen/Qwen3-coder-next). Use "baseline" for the retrieval baseline.')
    parser.add_argument('--mode', type=str, default="default", help='Prompting mode: default, plan, or code-plan. Default: default.')
    parser.add_argument('--provider', type=str, default="openai", help='Model provider: openai (default) or vllm (localhost:8000).')
    parser.add_argument('--dataset', type=str, help='Dataset to use. Possible options: ' + ', '.join(DatasetLoader.registered_datasets) + '.')
    parser.add_argument('--fewshot-dataset', type=str, help='Dataset to use to collect few-shot examples. Possible options: ' + ', '.join(DatasetLoader.registered_datasets) + '.', default="morehopqa")
    parser.add_argument('--strategy', type=str, help="Prompting strategy to use. Possible options: zeroshot, zeroshot-cot, 2-shot, 2-shot-cot, 3-shot, 3-shot-cot")
    parser.add_argument('--output_file', type=str, help='First part of the name of the output file. Will also include model, mode, strategy, dataset and timestamp. Default: output')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit evaluation to the first N samples (useful for quick tests).')

    args = parser.parse_args()

    if args.model is None or args.dataset is None or args.strategy is None:
        print("Missing arguments. Here are the possible options:")
        print("For --model: any OpenAI model name (e.g. gpt-4.1) or vLLM-served model name (e.g. Qwen/Qwen3-coder-next), or 'baseline'")
        print("For --mode: default, plan, code-plan")
        print("For --provider: openai, vllm")
        print("For --dataset: " + ", ".join(DatasetLoader.registered_datasets))
        print("For --strategy: zeroshot, zeroshot-cot, 2-shot, 2-shot-cot, 3-shot, 3-shot-cot")
        sys.exit(1)

    dataset = DatasetLoader.create(args.dataset)
    if args.max_samples is not None:
        dataset = DatasetSlice(dataset, args.max_samples)
    fewshot_dataset = DatasetLoader.create(args.fewshot_dataset)
    prompt_generator = PromptGenerator.create(args.strategy, fewshot_dataset)
    output_file_name = args.output_file if args.output_file is not None else "output"
    model = AbstractModel.create(
        model_name=args.model,
        output_file_name=output_file_name,
        prompt_generator=prompt_generator,
        mode=args.mode,
        provider=args.provider,
    )

    print(f"Using model: {args.model}")
    print(f"Using mode: {args.mode}")
    print(f"Using provider: {args.provider}")
    print(f"Using strategy: {args.strategy}")
    print(f"Using dataset: {args.dataset}")
    print(f"Using few-shot dataset: {args.fewshot_dataset}")
    print(f"Using output file: {args.output_file}")

    answers = model.get_answers_and_cache(dataset)
    if args.model == "baseline":
        postprocessed = postprocess_all_baseline(answers, dataset)
        results = evaluate_baseline(postprocessed)
    else:
        postprocessed = postprocess_all(answers, dataset)
        results = evaluate_all(postprocessed)

    # Sanitize model name for use in filename (replace / with -)
    model_slug = args.model.replace("/", "-")
    output_str = f"""

    Evaluation done. Results:
    - Model: {args.model}
    - Mode: {args.mode}
    - Provider: {args.provider}
    - Dataset: {args.dataset}
    - Strategy: {args.strategy}

    RESULT SUMMARY:
    - Total questions: {len(list(results.keys()))}
    - Correct answers in overall question: {[results[key]["case_1_em"] for key in results.keys()].count(True)}
    """
    for case_id in range(1,7):
        output_str += f"""
        - Avg precision (case_{case_id}): {sum(results[key][f"case_{case_id}_precision"] for key in results) / len(results):.3f}
        - Avg recall    (case_{case_id}): {sum(results[key][f"case_{case_id}_recall"] for key in results) / len(results):.3f}
        - Avg F1        (case_{case_id}): {sum(results[key][f"case_{case_id}_f1"] for key in results) / len(results):.3f}
        """

    print(output_str)

    output_file = f"results/{output_file_name}_{model_slug}_{args.mode}_{args.strategy}_{args.dataset}_{datetime.now().strftime('%y%m%d-%H%M%S')}.jsonl"
    i = 1
    original_output_file = output_file
    while os.path.exists(output_file):
        # don't overwrite
        print("WARNING: File already exists. Will not overwrite.")
        output_file = original_output_file + f"_{i}.jsonl"
        i += 1

    with open(output_file, "w") as f:
        for entry_id, entry_results in results.items():
            f.write(json.dumps({"_id": entry_id, **entry_results}) + "\n")

    print("Results written to file.")

if __name__ == '__main__':
    main()
