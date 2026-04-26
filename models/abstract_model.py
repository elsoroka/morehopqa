"""
Abstract model class to run evaluation. 
Should abstract away the prompt and loading of the model. 
Important: Cache all model answers.
"""

from abc import ABC, abstractmethod
from datetime import datetime


class AbstractModel(ABC):
    registered_models = ["gpt-3.5-turbo-direct", "gpt-4-turbo-direct", "gpt-4o-direct", "gpt-4o-plan", "gemma-7b", "llama-8b", "llama-70b", "mistral-7b", "baseline"]

    @abstractmethod
    def get_answers_and_cache(self, dataset) -> dict:
        """Should iterate over dataset and cache all answers.
        
        Returns: dict of answers (key: id in initial dataset, value: model_answer)"""
        pass

    @staticmethod
    def create(model_name, output_file_name, prompt_generator):
        if model_name in ("gpt-3.5-turbo-direct", "gpt-4-turbo-direct", "gpt-4o-direct"):
            from models.openai_direct_model import OpenAIDirectModel as cls
        elif model_name == "gpt-4o-plan":
            from models.openai_plan_model import OpenAIPlanModel as cls
        elif model_name == "gemma-7b":
            from models.gemma_7b import Gemma7B as cls
        elif model_name == "llama-8b":
            from models.llama_8b import Llama8b as cls
        elif model_name == "llama-70b":
            from models.llama_70b import Llama70b as cls
        elif model_name == "mistral-7b":
            from models.mistral_7b import Mistral7B as cls
        elif model_name == "baseline":
            from models.baseline import Baseline as cls
        else:
            raise ValueError(f"Model {model_name} not found.")

        return cls(
            model_name=model_name,
            output_file_name=f"{output_file_name}_{model_name}_{datetime.now().strftime('%y%m%d-%H%M%S')}.json",
            prompt_generator=prompt_generator,
        )