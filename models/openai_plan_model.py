"""
OpenAI model that plans its actions in natural language before answering.

Two-call approach per question:
  1. Plan call  – ask the model to outline what it needs to figure out.
  2. Answer call – inject the plan into the prompt and generate the final answer.

The cached entry gains a `<case_id>_plan` field alongside the usual
`<case_id>_prompt` and `<case_id>_answer` fields so the plans can be
inspected later.
"""

from models.abstract_model import AbstractModel
from openai import OpenAI
import json
from tqdm import tqdm

SYSTEM_PROMPT = """
You are a question answering system. The user will ask you a question and you will provide an answer.
You can generate as much text as you want to get to the solution. Your final answer must be contained in two brackets: <answer> </answer>.
"""

PLAN_SUFFIX = """
Before answering, write a concise step-by-step plan describing exactly what information you need to find and in what order. \
Do not answer the question yet — only output the plan.
"""

PLAN_INJECTION = """
Here is your plan for answering this question:
{plan}

Now follow the plan and answer the question.
"""

END_OF_PROMPT = """
If the answer is a date, format is as follows: YYYY-MM-DD (ISO standard)
If the answer is a name, format it as follows: Firstname Lastname
If the question is a yes or no question: answer with 'yes' or 'no' (without quotes)
If the answer contains any number, format it as a number, not a word, and only output that number.

Please provide the answer in the following format: <answer>*your answer here*</answer>
Answer as short as possible.
"""


class OpenAIPlanModel(AbstractModel):
    def __init__(self, model_name="gpt-4o-plan", output_file_name="output", provider="openai", prompt_generator=None):
        if provider == "openai":
            self.client = OpenAI()
        elif provider == "vllm":
            # TODO: Implement VLLM support
            pass
        else:
            raise ValueError(f"Invalid provider: {provider}")
        
        self.model_name = model_name.replace("-plan", "")
        self.output_file_name = output_file_name
        self.prompt_generator = prompt_generator

    def _call(self, user_content, max_tokens=1024):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
        )
        usage = response.usage
        return response.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens

    def get_plan(self, base_prompt):
        """First call: ask the model to plan, not answer."""
        return self._call(base_prompt + PLAN_SUFFIX, max_tokens=512)

    def get_answer(self, base_prompt, plan):
        """Second call: inject the plan and get the final answer."""
        augmented_prompt = base_prompt + PLAN_INJECTION.format(plan=plan) + END_OF_PROMPT
        return self._call(augmented_prompt, max_tokens=256)

    def get_prompt(self, question_entry, context, question):
        return self.prompt_generator.get_prompt(question_entry, context, question)

    def get_all_cases(self, entry):
        context = entry["context"]
        return {
            "case_1": self.get_prompt(entry, context, entry["question"]),
            "case_2": self.get_prompt(entry, context, entry["previous_question"]),
            "case_3": self.get_prompt(entry, context, entry["ques_on_last_hop"]),
            "case_4": self.get_prompt(entry, None,     entry["question_decomposition"][2]["question"]),
            "case_5": self.get_prompt(entry, context,  entry["question_decomposition"][1]["question"]),
            "case_6": self.get_prompt(entry, context,  entry["question_decomposition"][0]["question"]),
        }

    def get_answers_and_cache(self, dataset) -> dict:
        answers = dict()
        total_plan_tokens_in = 0
        total_plan_tokens_out = 0
        total_answer_tokens_in = 0
        total_answer_tokens_out = 0
        for entry in tqdm(dataset.items(), total=dataset.length):
            cases = self.get_all_cases(entry)
            answer_entry = {"_id": entry["_id"], "context": entry["context"]}

            for case_id, prompt in cases.items():
                plan, plan_tokens_in, plan_tokens_out = self.get_plan(prompt)
                answer, answer_tokens_in, answer_tokens_out = self.get_answer(prompt, plan)
                answer_entry[f"{case_id}_prompt"] = prompt
                answer_entry[f"{case_id}_plan"] = plan
                answer_entry[f"{case_id}_answer"] = answer
                answer_entry[f"{case_id}_plan_tokens_in"] = plan_tokens_in
                answer_entry[f"{case_id}_plan_tokens_out"] = plan_tokens_out
                answer_entry[f"{case_id}_answer_tokens_in"] = answer_tokens_in
                answer_entry[f"{case_id}_answer_tokens_out"] = answer_tokens_out
                total_plan_tokens_in += plan_tokens_in
                total_plan_tokens_out += plan_tokens_out
                total_answer_tokens_in += answer_tokens_in
                total_answer_tokens_out += answer_tokens_out

            answers[entry["_id"]] = answer_entry
            with open(f"models/cached_answers/{self.output_file_name}", "w") as f:
                json.dump(answers, f, indent=4)

        total_in = total_plan_tokens_in + total_answer_tokens_in
        total_out = total_plan_tokens_out + total_answer_tokens_out
        print(
            f"\nToken usage:"
            f"\n  Plan calls   — input: {total_plan_tokens_in:,}  output: {total_plan_tokens_out:,}"
            f"\n  Answer calls — input: {total_answer_tokens_in:,}  output: {total_answer_tokens_out:,}"
            f"\n  Total        — input: {total_in:,}  output: {total_out:,}  total: {total_in + total_out:,}"
        )
        return answers
