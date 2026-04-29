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
from models.openai_plan_model import OpenAIPlanModel
from openai import OpenAI
import json
from tqdm import tqdm
import re
import datetime
from postprocess import extract_and_parse_date

SYSTEM_PROMPT = """
You are a question answering system. The user will ask you a question and you will provide an answer.
You can generate as much text as you want to get to the solution. Your final answer must be contained in two brackets: <answer> </answer>.
"""

PLAN_CONTEXT_PREFIX = "Context:\n#CONTEXT\n\n"

PLAN_QUESTION_PREFIX = "Question: #QUESTION\n\n"

PLAN_SUFFIX = """
Write a concise step-by-step plan as a Python function `answer_question()->str:` to answer the above question. You have access to the following:
* a large language model you can call with the function `llm.prompt(prompt:str)->str:`
* a string variable `question` that contains the exact question you need to answer
* a string variable `context` that contains the exact context you were provided to answer the question.
* a function `clean_date(date_str:str)->datetime.datetime:` that will extract and parse a date string into a datetime.datetime object. Always use clean_date for date parsing — never use datetime.strptime(...).date() or other methods that return datetime.date, as mixing datetime.date and datetime.datetime causes errors.
* a function `clean_answer(answer:str)->str:` that wraps text in <answer>...</answer> tags, normalizing any existing tags if present.
* all built-in Python functions and libraries

Hints:
* Break the question into smaller steps.
* Your plan must return the final answer wrapped in <answer>...</answer> tags using clean_answer.
* Do not assume llm.prompt() can return valid JSON.
* Remember that the LLM will only see the prompt you write, so write your prompts carefully and include all necessary context to solve the step. You can use the `context` and `question` variables to help you write prompts for `llm.prompt()`.
* Remember that llm.prompt() can only return text output, so if you need any other data type you must convert it yourself.
* Always use clean_date for date parsing. Never call .date() on the result — keep everything as datetime.datetime for arithmetic.

Example question: What is three days after the last date mentioned in the context?
Example plan:
```python
def answer_question()->str:
    import datetime, re
    def strip_tags(text):
        import re
        m = re.search(r'<answer>(.*?)</answer>', text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else text.strip()
    last_date_str = strip_tags(llm.prompt(f"Find the last date mentioned in this text and return it as a string formatted as YYYY-MM-DD (ISO standard):\\nText:\\n{context}"))
    last_date = clean_date(last_date_str)
    three_days_after = last_date + datetime.timedelta(days=3)
    return clean_answer(three_days_after.strftime('%Y-%m-%d'))
```
Do not answer the question yourself — only output the plan as a Python code block.
"""

END_OF_PROMPT = """
If the answer is a date, your code should format it as a YYYY-MM-DD (ISO standard) string.
If the answer is a name, your code should format it as Firstname Lastname.
If the question is a yes or no question: your code should return 'yes' or 'no' (without quotes).
If the answer contains any number, your code should format it as a string of digits.

Your code should call clean_answer on the final answer string to wrap it in <answer>...</answer> tags.
"""

def _lint_plan(code: str) -> str | None:
    """Return an error string if the plan fails structural checks, else None."""
    import ast as _ast
    try:
        tree = _ast.parse(code)
    except SyntaxError as e:
        return str(e)
    func_defs = {node.name: node for node in _ast.walk(tree) if isinstance(node, _ast.FunctionDef)}
    if 'answer_question' not in func_defs:
        return "Lint error: `answer_question` function is not defined"
    fn = func_defs['answer_question']
    if not any(isinstance(n, _ast.Return) and n.value is not None for n in _ast.walk(fn)):
        return "Lint error: `answer_question` function has no return statement"
    return None


def safe_parse_int(s: str) -> int:
    try:
        return int(s)
    except Exception:
        pass
    s = ''.join([c for c in s if c.isdigit() or c == '-'])
    try:
        return int(s)
    except Exception:
        return float('nan')

def safe_parse_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        pass
    s = ''.join([c for c in s if c.isdigit() or c == '.' or c == '-'])
    try:
        return float(s)
    except Exception:
        return float('nan')


class OpenAICodePlanModel(AbstractModel):
    def __init__(self, model_name="gpt-4.1", output_file_name="output", provider="openai", prompt_generator=None):
        if provider == "openai":
            self.client = OpenAI()
        elif provider == "vllm":
            self.client = OpenAI(base_url="http://localhost:8000/v1", api_key="placeholder")
        elif provider == "stanford":
            import os
            self.client = OpenAI(base_url="https://aiapi-prod.stanford.edu/v1", api_key=os.getenv("STANFORD_API_KEY"))
        else:
            raise ValueError(f"Invalid provider: {provider}")
        self.model_name = model_name
        self.output_file_name = output_file_name
        self.provider = provider
        self.prompt_generator = prompt_generator
        # Per-case token counters and sub-call log, updated by self.prompt() during exec
        self._answer_tokens_in = 0
        self._answer_tokens_out = 0
        self._sub_calls = []  # list of {"prompt": ..., "response": ...} per case

    def clean_answer(self, answer: str) -> str:
        """Wrap answer in <answer> tags, extracting inner content first if tags are already present."""
        m = re.search(r'<answer>(.*?)</answer>', answer, re.IGNORECASE | re.DOTALL)
        content = m.group(1).strip() if m else answer.strip()
        return f"<answer>{content}</answer>"

    def _call(self, user_content, max_tokens=1024, system_prompt=SYSTEM_PROMPT):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
        )
        usage = response.usage
        return response.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens

    def _build_plan_prompt(self, context, question):
        """Build a plan-only prompt with no answer-format instructions."""
        
        if not hasattr(self, "nlp_plan"):
            other_planner_model = OpenAIPlanModel(model_name=self.model_name, output_file_name=self.output_file_name, provider=self.provider, prompt_generator=self.prompt_generator)
            self.nlp_plan = other_planner_model.get_plan(context, question)
        
        prompt = ""
        if context is not None:
            context_string = ""
            for i, para in enumerate(context):
                context_string += f"\n{i+1}: {para[0]}\n{' '.join(para[1])}"
            prompt += PLAN_CONTEXT_PREFIX.replace("#CONTEXT", context_string)
        prompt += PLAN_QUESTION_PREFIX.replace("#QUESTION", question)
        prompt += PLAN_SUFFIX + f"Outline of your code:\n{self.nlp_plan}\n" + END_OF_PROMPT
        return prompt

    def get_plan(self, context, question, max_tries=3, prior_execution_error=None):
        """First call: ask the model to write a Python plan."""
        error_message = None
        plan_tokens_in = 0
        plan_tokens_out = 0
        planner_prompt = None
        plan = None

        for _ in range(max_tries):
            planner_prompt = self._build_plan_prompt(context, question)
            if prior_execution_error:
                planner_prompt += f"\nThe previous plan raised a runtime error during execution: {prior_execution_error}\nPlease rewrite the plan to fix this error."
            if error_message:
                planner_prompt += f"\nSyntax error from previous attempt: {error_message}\nPlease fix the code and try again."

            raw, tok_in, tok_out = self._call(planner_prompt, max_tokens=2048, system_prompt=None)
            plan_tokens_in += tok_in
            plan_tokens_out += tok_out
            #print("Raw plan:", raw)

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
            except SyntaxError as e:
                error_message = str(e)
                continue
            lint_error = _lint_plan(plan)
            if lint_error:
                error_message = lint_error
                continue
            return plan, plan_tokens_in, plan_tokens_out, planner_prompt, True

        return plan, plan_tokens_in, plan_tokens_out, planner_prompt, False

    def prompt(self, prompt: str) -> str:
        """Called from within exec'd plan code to invoke the LLM on a sub-task.

        Instructs the model to wrap its answer in <answer> tags and retries up
        to 3 times if the tags are absent (helps small models that mix
        reasoning with output).
        """
        tag_instruction = "\n\nEnclose your final answer in <answer>...</answer> tags."
        current_prompt = prompt + tag_instruction
        response = None
        for attempt in range(3):
            response, tok_in, tok_out = self._call(current_prompt, max_tokens=2048, system_prompt=None)
            self._answer_tokens_in += tok_in
            self._answer_tokens_out += tok_out
            self._sub_calls.append({"prompt": current_prompt, "response": response,
                                     "tokens_in": tok_in, "tokens_out": tok_out})
            m = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
            if m:
                return f"<answer>{m.group(1).strip()}</answer>"
            current_prompt = (prompt + tag_instruction +
                              f"\n\nAttempt {attempt + 1} failed: you must wrap your answer in <answer>...</answer> tags. Try again.")
        # Fallback: return raw response so the plan can still try to use it
        return response

    def get_prompt(self, question_entry, context, question):
        return self.prompt_generator.get_prompt(question_entry, context, question)

    def get_all_cases(self, entry):
        """Return a dict mapping case_id -> (context, question) for all 6 cases."""
        selected = getattr(self, 'cases', None)
        context = entry["context"]
        q = entry["question_decomposition"]
        all_cases = {
            "case_1": (context, entry["question"]),
            "case_2": (context, entry["previous_question"]),
            "case_3": (context, entry["ques_on_last_hop"]),
            "case_4": (None,    q[2]["question"]),
            "case_5": (context, q[1]["question"]),
            "case_6": (context, q[0]["question"]),
        }
        if selected is None:
            return all_cases
        return {k: v for k, v in all_cases.items() if k in selected}

    def get_answers_and_cache(self, dataset, max_exec_retries=3) -> dict:
        answers = dict()
        total_plan_tokens_in = 0
        total_plan_tokens_out = 0
        total_answer_tokens_in = 0
        total_answer_tokens_out = 0
    
        for entry in tqdm(dataset.items(), total=dataset.length):
            cases = self.get_all_cases(entry)
            
            answer_entry = {"_id": entry["_id"], "context": entry["context"]}
            all_correct_answers = {
                "case_1": entry["answer"],
                "case_2": entry["previous_answer"],
                "case_3": entry["answer"],
                "case_4": entry["answer"],
                "case_5": entry["previous_answer"],
                "case_6": entry["question_decomposition"][0]["answer"],
            }
            correct_answers = {k: v for k, v in all_correct_answers.items() if k in cases}
            for case_id, (context, question) in cases.items():
                self._answer_tokens_in = 0
                self._answer_tokens_out = 0
                self._sub_calls = []

                exec_error = None
                result = None
                plan = None
                planner_prompt = None
                valid_plan = False
                case_plan_tokens_in = 0
                case_plan_tokens_out = 0

                for exec_attempt in range(max_exec_retries):
                    plan, plan_tokens_in, plan_tokens_out, planner_prompt, valid_plan = self.get_plan(
                        context, question, prior_execution_error=exec_error
                    )
                    case_plan_tokens_in += plan_tokens_in
                    case_plan_tokens_out += plan_tokens_out
                    total_plan_tokens_in += plan_tokens_in
                    total_plan_tokens_out += plan_tokens_out

                    if not valid_plan:
                        result = None
                        break

                    # llm/question/context must be in the globals dict so they
                    # are visible inside nested functions defined by the plan.
                    exec_globals = {
                        "clean_answer": self.clean_answer,
                        "llm": self,
                        "question": question,
                        "context": '\n\n'.join({f"{c[0]}\n{' '.join(c[1])}" for c in entry["context"]}),
                        'clean_date': extract_and_parse_date,
                        "int": safe_parse_int,
                        "float": safe_parse_float,
                        "datetime": datetime,
                        "re": re,
                    }
                    local_vars = {}
                    try:
                        exec(plan, exec_globals, local_vars)
                        result = local_vars.get("result")
                        if callable(result):
                            print(f"Exec produced a callable for {case_id}, treating as None")
                            result = None
                        break  # success — stop retrying
                    except Exception as e:
                        exec_error = str(e)
                        print(f"Exec error for {case_id} (attempt {exec_attempt + 1}/{max_exec_retries}): {e}")
                        result = None

                answer_entry[f"{case_id}_prompt"] = planner_prompt
                answer_entry[f"{case_id}_plan"] = plan
                answer_entry[f"{case_id}_planner_prompt"] = planner_prompt
                answer_entry[f"{case_id}_plan_tokens_in"] = case_plan_tokens_in
                answer_entry[f"{case_id}_plan_tokens_out"] = case_plan_tokens_out
                answer_entry[f"{case_id}_answer"] = result
                answer_entry[f"{case_id}_sub_calls"] = self._sub_calls
                answer_entry[f"{case_id}_answer_tokens_in"] = self._answer_tokens_in
                answer_entry[f"{case_id}_answer_tokens_out"] = self._answer_tokens_out
                total_answer_tokens_in += self._answer_tokens_in
                total_answer_tokens_out += self._answer_tokens_out

            answers[entry["_id"]] = answer_entry
            with open(f"models/cached_answers/{self.output_file_name}", "w") as f:
                json.dump(answers, f, indent=4, default=lambda o: f"<non-serializable: {type(o).__name__}>")

        total_in = total_plan_tokens_in + total_answer_tokens_in
        total_out = total_plan_tokens_out + total_answer_tokens_out
        print(
            f"\nToken usage:"
            f"\n  Plan calls     — input: {total_plan_tokens_in:,}  output: {total_plan_tokens_out:,}"
            f"\n  Executor calls — input: {total_answer_tokens_in:,}  output: {total_answer_tokens_out:,}"
            f"\n  Total          — input: {total_in:,}  output: {total_out:,}  total: {total_in + total_out:,}"
        )
        return answers
