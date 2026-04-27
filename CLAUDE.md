# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

All scripts must be run from the `morehopqa/` directory (paths are relative to it).

```bash
cd morehopqa
python3 -m venv .venv
source .venv/bin/activate
pip install openai spacy tqdm numerizer python-dateutil python-dotenv
python3 -m spacy download en_core_web_sm
```

Place a `.env` file in `morehopqa/` with your API key — it is loaded automatically at startup:
```
OPENAI_API_KEY=sk-...
```

## Commands

**Run evaluation for a single model/strategy:**
```bash
cd morehopqa
.venv/bin/python3 run_evaluation.py --model gpt-4.1 --mode default --dataset morehopqa --strategy zeroshot --output_file test
```

**Quick test on a few samples:**
```bash
.venv/bin/python3 run_evaluation.py --model gpt-4.1 --mode plan --dataset morehopqa --strategy zeroshot --output_file test --max-samples 3
```

**Run with a vLLM-served model (assumes model running on localhost:8000):**
```bash
.venv/bin/python3 run_evaluation.py --model Qwen/Qwen3-coder-next --mode code-plan --provider vllm --dataset morehopqa --strategy zeroshot --output_file test
```

**Run all models from the paper:**
```bash
cd morehopqa
bash run_evaluation.sh
```

**Summarize results:**
Open `morehopqa/summarize_results.ipynb` in Jupyter.

**Available options:**
- `--model`: any OpenAI model name (e.g. `gpt-4.1`, `gpt-4o`) or vLLM-served model name (e.g. `Qwen/Qwen3-coder-next`), or `baseline`
- `--mode`: `default` (direct answer), `plan` (NL plan then answer), `code-plan` (generate + execute Python plan). Default: `default`
- `--provider`: `openai` (default) or `vllm` (localhost:8000)
- `--dataset`: `morehopqa` (1118 samples), `morehopqa-150` (150-sample subset)
- `--strategy`: `zeroshot`, `zeroshot-cot`, `2-shot`, `2-shot-cot`, `3-shot`, `3-shot-cot`
- `--max-samples N`: limit to the first N samples (testing only)

## Architecture

Each evaluation run flows through four stages:

1. **Dataset loading** (`datasets/`) — `DatasetLoader.create(name)` returns a loader that yields dataset entries. Each entry has 6 question variants per sample (see below).

2. **Prompt generation** (`models/prompt_generator.py`) — `PromptGenerator.create(strategy, dataset)` returns a generator that formats prompts with optional context paragraphs and few-shot examples. Few-shot examples are sampled by matching `answer_type` and `previous_answer_type`.

3. **Model inference** (`models/`) — Each model subclass implements `get_answers_and_cache(dataset)`, which iterates the dataset, runs all 6 cases per entry, and writes intermediate results incrementally to `models/cached_answers/<output_file>.json`.

4. **Postprocessing + evaluation** (`postprocess.py`, `evaluate.py`) — `postprocess_all` normalizes model outputs by answer type (string/person/org → tag extraction; number/year → float normalization; date/datetime → ISO parsing with NER fallback). `evaluate_all` then computes exact match and F1 for each of the 6 cases.

Final results are written to `results/<output_file>_<model>_<mode>_<strategy>_<dataset>_<timestamp>.jsonl`.

### The 6 evaluation cases per sample

Each dataset entry contains a new generative question layered on top of an existing multi-hop question:

| Case | Question | Answer |
|------|----------|--------|
| case_1 | The new MoreHopQA question | `answer` |
| case_2 | The original multi-hop question (`previous_question`) | `previous_answer` |
| case_3 | `ques_on_last_hop` (composite path to final answer) | `answer` |
| case_4 | `question_decomposition[2]` (last sub-question, no context) | `answer` |
| case_5 | `question_decomposition[1]` | `previous_answer` |
| case_6 | `question_decomposition[0]` (first sub-question) | `question_decomposition[0].answer` |

The `baseline` model only evaluates cases 1 and 2.

### Plan-then-answer mode (`--mode plan`)

`models/openai_plan_model.py` makes two API calls per question: a planning call that asks the model to outline its reasoning steps without answering, then an answer call with the plan injected into the prompt. The cache gains a `<case_id>_plan` field alongside the usual prompt/answer fields.

### Code-plan mode (`--mode code-plan`)

`models/openai_code_plan_model.py` asks the model to write a Python function `answer_question()` that solves the question by calling back into the LLM via `llm.prompt()` for sub-tasks, then executes the generated code. The cache gains `<case_id>_plan`, `<case_id>_planner_prompt`, and `<case_id>_sub_calls` (a list of every `llm.prompt()` call made during execution).

**Note:** gpt-4o performs poorly with both planning modes — its built-in reasoning interferes with the structured planning format. Use `gpt-4.1` instead; it follows the plan format faithfully and shows meaningfully better results. Planner evaluations should use `zeroshot` strategy only — few-shot examples anchor the model toward direct NL answers and undermine the planning approach.

### Adding a new model

Subclass `AbstractModel`, implement `get_answers_and_cache(dataset) -> dict`, and add a branch for it in `AbstractModel.create()` (lazy import) and `AbstractModel.registered_models`.
