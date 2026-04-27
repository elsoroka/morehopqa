"""
OpenAI model that plans its actions as executable Python code, then runs that
code to produce an answer.  Sub-tasks are delegated back to the same LLM via
``llm.prompt()``.

Three-stage approach per question:
  1. Plan call  – ask the model to write a Python function ``answer_question()``.
  2. Exec stage – run the generated code; ``llm.prompt()`` calls are made as
     needed from within the plan.
  3. Cache       – store the plan, all sub-prompts, and the final answer.

The cached entry gains a ``<case_id>_plan`` field alongside the usual
``<case_id>_prompt`` and ``<case_id>_answer`` fields so the plans can be
inspected later.
"""

from models.abstract_model import AbstractModel
from openai import OpenAI
import json
from tqdm import tqdm
import re

SYSTEM_PROMPT = """
You are a question answering system. The user will ask you a question and you will provide an answer.
You can generate as much text as you want to get to the solution. Your final answer must be contained in two brackets: <answer> </answer>.
"""

PLAN_SUFFIX = """
Write a concise step-by-step plan as a Python function `answer_question()->str:` to answer the following question. You have access to the following:
* a large language model you can call with the function `llm.prompt(prompt:str)->str:`
* a string variable `question` that contains the exact question you need to answer
* a string variable`context` that contains the exact context you were provided to answer the question.
* a function clean_answer(answer:str)->str: that will clean your final answer text and ensure it is wrapped in <answer> </answer> tags.
* all built-in Python functions and libraries

Hints:
* Break the question into smaller steps.
* Your plan must return the answer as a string wrapped in <answer> </answer> tags.
* Do not assume llm.prompt() can return valid JSON.
* Remember that the LLM will only see the prompt you write, so write your prompts carefully and include all necessary context to solve the step. You can use the `context` and `question` variables to help you write prompts for `llm.prompt()`.
* Remember that llm.prompt() can only return text output, so if you need any other data type you must convert it yourself.
* llm.prompt() may return text that already contains <answer> </answer> tags. Always strip them before using the value, and add them yourself only in the final return statement.

Example question: What is three days after the last date mentioned in the context?
Example plan:
```python
def answer_question()->str:
    import datetime, re
    def strip_answer_tags(text):
        m = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else text.strip()
    last_date_str = strip_answer_tags(llm.prompt(f"Find the last date mentioned in this text and return it as a string formatted as YYYY-MM-DD (ISO standard):\\nText:\\n{context}"))
    last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
    three_days_after = last_date + datetime.timedelta(days=3)
    return clean_answer(three_days_after.strftime('%Y-%m-%d'))
```
Do not answer the question yourself — only output the plan as a Python code block.
"""

END_OF_PROMPT = """
If the answer is a date, your code should format it as a YYYY-MM-DD (ISO standard) string
If the answer is a name, your code should format it as Firstname Lastname
If the question is a yes or no question: your code should answer with 'yes' or 'no' (without quotes)
If the answer contains any number, your code should format it as a string of digits.

Your code should call the clean_answer function to ensure the answer does not contain extra formatting and is wrapped in <answer> </answer> tags.
"""


class OpenAICodePlanModel(AbstractModel):
    def __init__(self, model_name="gpt-4o-code-plan", output_file_name="output", provider="openai", prompt_generator=None):
        if provider == "openai":
            self.client = OpenAI()
        elif provider == "vllm":
            # TODO: Implement VLLM support
            pass
        else:
            raise ValueError(f"Invalid provider: {provider}")

        self.model_name = model_name.replace("-code-plan", "")
        self.output_file_name = output_file_name
        self.prompt_generator = prompt_generator
        # Per-case token counters and sub-call log, updated by self.prompt() during exec
        self._answer_tokens_in = 0
        self._answer_tokens_out = 0
        self._sub_calls = []  # list of {"prompt": ..., "response": ...} per case

    def clean_answer(self, answer: str) -> str:
        m = re.search(r'<answer>(.*?)</answer>', answer, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else answer.strip()

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

    def get_plan(self, base_prompt, max_tries=3):
        """First call: ask the model to write a Python plan."""
        error_message = None
        plan_tokens_in = 0
        plan_tokens_out = 0
        planner_prompt = None
        plan = None

        for _ in range(max_tries):
            planner_prompt = base_prompt + PLAN_SUFFIX + END_OF_PROMPT
            if error_message:
                planner_prompt += f"\nError from previous attempt: {error_message}\nPlease fix the code and try again."

            raw, tok_in, tok_out = self._call(planner_prompt, max_tokens=512)
            plan_tokens_in += tok_in
            plan_tokens_out += tok_out
            print("Raw plan:", raw)

            # Strip markdown code fences if present
            stripped = raw.strip()
            if stripped.startswith("```python"):
                stripped = stripped[len("```python"):].lstrip("\n")
            elif stripped.startswith("```"):
                stripped = stripped[3:].lstrip("\n")
            if stripped.endswith("```"):
                stripped = stripped[:-3].rstrip()
            plan = stripped

            # Append the module-level call that captures the return value.
            # Check only unindented lines so we don't mistake `result = ...`
            # inside the function body for the top-level invocation.
            last_unindented = next(
                (l for l in reversed(plan.splitlines())
                 if l and not l[0].isspace()),
                ""
            )
            if not ("answer_question()" in last_unindented and last_unindented.strip().startswith("result")):
                plan = plan + "\nresult = answer_question()"

            try:
                compile(plan, "<string>", "exec")
                return plan, plan_tokens_in, plan_tokens_out, planner_prompt, True
            except SyntaxError as e:
                error_message = str(e)

        return plan, plan_tokens_in, plan_tokens_out, planner_prompt, False

    def prompt(self, prompt: str) -> str:
        """Called from within exec'd plan code to invoke the LLM on a sub-task."""
        response, tok_in, tok_out = self._call(prompt, max_tokens=512)
        self._answer_tokens_in += tok_in
        self._answer_tokens_out += tok_out
        self._sub_calls.append({"prompt": prompt, "response": response,
                                 "tokens_in": tok_in, "tokens_out": tok_out})
        return response

    def get_prompt(self, question_entry, context, question):
        return self.prompt_generator.get_prompt(question_entry, context, question)

    def get_all_cases(self, entry):
        """Return a dict mapping case_id -> (prompt, question) for all 6 cases."""
        context = entry["context"]
        q = entry["question_decomposition"]
        return {
            "case_1": (self.get_prompt(entry, context, entry["question"]),                entry["question"]),
            "case_2": (self.get_prompt(entry, context, entry["previous_question"]),       entry["previous_question"]),
            "case_3": (self.get_prompt(entry, context, entry["ques_on_last_hop"]),        entry["ques_on_last_hop"]),
            "case_4": (self.get_prompt(entry, None,    q[2]["question"]),                 q[2]["question"]),
            "case_5": (self.get_prompt(entry, context, q[1]["question"]),                 q[1]["question"]),
            "case_6": (self.get_prompt(entry, context, q[0]["question"]),                 q[0]["question"]),
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

            for case_id, (prompt, question) in cases.items():
                self._answer_tokens_in = 0
                self._answer_tokens_out = 0
                self._sub_calls = []

                plan, plan_tokens_in, plan_tokens_out, planner_prompt, valid_plan = self.get_plan(prompt)
                total_plan_tokens_in += plan_tokens_in
                total_plan_tokens_out += plan_tokens_out

                answer_entry[f"{case_id}_prompt"] = prompt
                answer_entry[f"{case_id}_plan"] = plan
                answer_entry[f"{case_id}_planner_prompt"] = planner_prompt
                answer_entry[f"{case_id}_plan_tokens_in"] = plan_tokens_in
                answer_entry[f"{case_id}_plan_tokens_out"] = plan_tokens_out

                if not valid_plan:
                    answer_entry[f"{case_id}_answer"] = None
                    answer_entry[f"{case_id}_answer_tokens_in"] = 0
                    answer_entry[f"{case_id}_answer_tokens_out"] = 0
                else:
                    # llm/question/context must be in the globals dict so they
                    # are visible inside nested functions defined by the plan.
                    exec_globals = {
                        "clean_answer": self.clean_answer,
                        "llm": self,
                        "question": question,
                        "context": entry["context"],
                    }
                    local_vars = {}
                    try:
                        exec(plan, exec_globals, local_vars)
                        result = local_vars.get("result")
                    except Exception as e:
                        print(f"Exec error for {case_id}: {e}")
                        result = None

                    print("Result of execution:", result)
                    answer_entry[f"{case_id}_answer"] = result
                    answer_entry[f"{case_id}_sub_calls"] = self._sub_calls
                    answer_entry[f"{case_id}_answer_tokens_in"] = self._answer_tokens_in
                    answer_entry[f"{case_id}_answer_tokens_out"] = self._answer_tokens_out
                    total_answer_tokens_in += self._answer_tokens_in
                    total_answer_tokens_out += self._answer_tokens_out

            answers[entry["_id"]] = answer_entry
            with open(f"models/cached_answers/{self.output_file_name}", "w") as f:
                json.dump(answers, f, indent=4)

        total_in = total_plan_tokens_in + total_answer_tokens_in
        total_out = total_plan_tokens_out + total_answer_tokens_out
        print(
            f"\nToken usage:"
            f"\n  Plan calls     — input: {total_plan_tokens_in:,}  output: {total_plan_tokens_out:,}"
            f"\n  Executor calls — input: {total_answer_tokens_in:,}  output: {total_answer_tokens_out:,}"
            f"\n  Total          — input: {total_in:,}  output: {total_out:,}  total: {total_in + total_out:,}"
        )
        return answers
