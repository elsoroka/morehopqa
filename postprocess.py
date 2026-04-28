import re
import string
from collections import Counter
from datetime import datetime
from dateutil import parser
import spacy
import numerizer
from copy import deepcopy
from tqdm import tqdm
from datasets.abstract_dataset_loader import DatasetLoader


nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])


def extract_and_parse_date(date_str):
      # Clean the string by removing non-date words and extracting potential date ranges
      clean_date_str = re.sub(r"(born on|born|\bto\b)", "", date_str).strip()
      # Try to parse the date
      default_date = datetime(datetime.now().year, 1, 1)
      parsed_date = parser.parse(clean_date_str, fuzzy=True, default=default_date)
      return parsed_date


def parse_answer_tags(answer):
    """Return text content between <answer> and </answer> tags."""
    if answer is None:
        return None
    try:
        return re.match(r'.*<answer>(.*)</answer>.*', answer, re.IGNORECASE | re.DOTALL).group(1).strip()
    except AttributeError:
        return answer


def postprocess_date(answer):
    """Compare two date answers. Try parsing the date and then compare.

    If the date is not in the right format, try NER to find the date in the text."""
    if answer is None:
        return None
    try:
        model_date = extract_and_parse_date(answer)
        return model_date.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        # Try to use NER to find the date in the text
        ner = nlp(answer)
        date_ent = None
        for ent in ner.ents:
            if ent.label_ == 'DATE':
                date_ent = ent
        try:
            model_date = extract_and_parse_date(date_ent.text)
            return model_date.strftime("%Y-%m-%d %H:%M")
        except Exception as e:
            return answer
        

def postprocess_number(answer):
    if answer is None:
        return None
    try:
        return str(float(answer.replace(",", "")))
    except ValueError:
        # Try to use numerizer to find the date in the text
        ner = nlp(answer)
        num_ent = None
        try:
            num_ent = list(ner._.numerize().items())[-1][1]
            return str(float(num_ent))
        except Exception as e:
            try:
                for ent in answer.split():
                    try:
                        model_number = float(ent)
                    except:
                        continue
                return str(model_number)
            except Exception:
                return answer


def postprocess_baseline(model_answer, ground_truth_answer):
    """Postprocess the model answer to match the ground truth answer."""
    res_entry = dict()
    if ground_truth_answer['answer_type'] == 'string' or ground_truth_answer['answer_type'] == 'letter' or ground_truth_answer['answer_type'] == 'person' or ground_truth_answer['answer_type'] == 'organization'  or ground_truth_answer['answer_type'] == 'character':
        res_entry["case_1_pred_extr"] = parse_answer_tags(model_answer["case_1_answer"])
        res_entry["case_1_ground_truth"] = ground_truth_answer["answer"]
        
    elif ground_truth_answer['answer_type'] == 'number' or ground_truth_answer['answer_type'] == 'year':
        res_entry["case_1_pred_extr"] = postprocess_number(parse_answer_tags(model_answer["case_1_answer"]))
        res_entry["case_1_ground_truth"] = postprocess_number(ground_truth_answer["answer"])

    elif ground_truth_answer['answer_type'] == 'date' or ground_truth_answer['answer_type'] == 'datetime':
        res_entry["case_1_pred_extr"] = postprocess_date(parse_answer_tags(model_answer["case_1_answer"]))
        res_entry["case_1_ground_truth"] = postprocess_date(ground_truth_answer["answer"])
    else:
        raise ValueError(f"Answer type {ground_truth_answer['answer_type']} not supported.")
                         
    if ground_truth_answer['previous_answer_type'] == 'person' or ground_truth_answer['previous_answer_type'] == 'place' or ground_truth_answer['previous_answer_type'] == 'organization':
        res_entry["case_2_pred_extr"] = parse_answer_tags(model_answer["case_2_answer"])
        res_entry["case_2_ground_truth"] = ground_truth_answer["previous_answer"]
    elif ground_truth_answer['previous_answer_type'] == 'number' or ground_truth_answer['previous_answer_type'] == 'year':
        res_entry["case_2_pred_extr"] = postprocess_number(model_answer["case_2_answer"])
        res_entry["case_2_ground_truth"] = postprocess_number(ground_truth_answer["previous_answer"])
    elif ground_truth_answer['previous_answer_type'] == 'date' or ground_truth_answer['previous_answer_type'] == 'datetime':
        res_entry["case_2_pred_extr"] = postprocess_date(model_answer["case_2_answer"])
        res_entry["case_2_ground_truth"] = postprocess_date(ground_truth_answer["previous_answer"])
    else:
        raise ValueError(f"Previous answer type {ground_truth_answer['previous_answer_type']} not supported.")

    return res_entry


def postprocess(model_answer, ground_truth_answer, cases=None):
    """Postprocess the model answer to match the ground truth answer."""
    if cases is None:
        cases = {f"case_{i}" for i in range(1, 7)}
    res_entry = dict()

    # Cases 1, 3, 4 share the same answer and answer_type
    cases_134 = [c for c in ["case_1", "case_3", "case_4"] if c in cases]
    if cases_134:
        if ground_truth_answer['answer_type'] in ('string', 'letter', 'person', 'organization', 'character'):
            for c in cases_134:
                res_entry[f"{c}_pred_extr"] = parse_answer_tags(model_answer[f"{c}_answer"])
                res_entry[f"{c}_ground_truth"] = ground_truth_answer["answer"]
        elif ground_truth_answer['answer_type'] in ('number', 'year'):
            for c in cases_134:
                res_entry[f"{c}_pred_extr"] = postprocess_number(parse_answer_tags(model_answer[f"{c}_answer"]))
                res_entry[f"{c}_ground_truth"] = postprocess_number(ground_truth_answer["answer"])
        elif ground_truth_answer['answer_type'] in ('date', 'datetime'):
            for c in cases_134:
                res_entry[f"{c}_pred_extr"] = postprocess_date(parse_answer_tags(model_answer[f"{c}_answer"]))
                res_entry[f"{c}_ground_truth"] = postprocess_date(ground_truth_answer["answer"])
        else:
            raise ValueError(f"Answer type {ground_truth_answer['answer_type']} not supported.")

    # Cases 2, 5 share the same answer and previous_answer_type
    cases_25 = [c for c in ["case_2", "case_5"] if c in cases]
    if cases_25:
        if ground_truth_answer['previous_answer_type'] in ('person', 'place', 'organization'):
            for c in cases_25:
                res_entry[f"{c}_pred_extr"] = parse_answer_tags(model_answer[f"{c}_answer"])
                res_entry[f"{c}_ground_truth"] = ground_truth_answer["previous_answer"]
        elif ground_truth_answer['previous_answer_type'] in ('number', 'year'):
            for c in cases_25:
                res_entry[f"{c}_pred_extr"] = postprocess_number(parse_answer_tags(model_answer[f"{c}_answer"]))
                res_entry[f"{c}_ground_truth"] = postprocess_number(ground_truth_answer["previous_answer"])
        elif ground_truth_answer['previous_answer_type'] in ('date', 'datetime'):
            for c in cases_25:
                res_entry[f"{c}_pred_extr"] = postprocess_date(parse_answer_tags(model_answer[f"{c}_answer"]))
                res_entry[f"{c}_ground_truth"] = postprocess_date(ground_truth_answer["previous_answer"])
        else:
            raise ValueError(f"Previous answer type {ground_truth_answer['previous_answer_type']} not supported.")

    # Case 6 is standalone
    if "case_6" in cases:
        res_entry["case_6_pred_extr"] = parse_answer_tags(model_answer["case_6_answer"])
        res_entry["case_6_ground_truth"] = ground_truth_answer["question_decomposition"][0]["answer"]

    return res_entry


def postprocess_all_baseline(model_answers: dict, dataset: DatasetLoader):
    res = dict()
    data = dataset.items()
    for entry in tqdm(data, total=dataset.length):
        result_entry = deepcopy(entry)
        result_entry["_id"] = entry["_id"]
        result_entry.update(model_answers[entry["_id"]])
        result_entry.update(postprocess_baseline(model_answers[entry["_id"]], entry))
        res[result_entry["_id"]] = result_entry
    return res


def postprocess_all(model_answers: dict, dataset: DatasetLoader, cases=None):
    """Evaluate all answers from the model and compare them to the ground truth."""
    res = dict()
    data = dataset.items()
    for entry in tqdm(data, total=dataset.length):
        result_entry = deepcopy(entry)
        result_entry["_id"] = entry["_id"]
        result_entry.update(model_answers[entry["_id"]])
        result_entry.update(postprocess(model_answers[entry["_id"]], entry, cases=cases))
        res[result_entry["_id"]] = result_entry
    return res
