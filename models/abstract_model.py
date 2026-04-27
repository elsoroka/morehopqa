"""
Abstract model class to run evaluation.
Should abstract away the prompt and loading of the model.
Important: Cache all model answers.
"""

from abc import ABC, abstractmethod
from datetime import datetime


class AbstractModel(ABC):
    registered_models = ["gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o", "gpt-4.1", "gemma-7b", "llama-8b", "llama-70b", "mistral-7b", "baseline"]

    @abstractmethod
    def get_answers_and_cache(self, dataset) -> dict:
        """Should iterate over dataset and cache all answers.

        Returns: dict of answers (key: id in initial dataset, value: model_answer)"""
        pass

    @staticmethod
    def create(model_name, output_file_name, prompt_generator, mode="default", provider="openai"):
        model_slug = model_name.replace("/", "-")
        stamped_output_file = f"{output_file_name}_{model_slug}_{mode}_{datetime.now().strftime('%y%m%d-%H%M%S')}.json"

        if model_name == "baseline":
            from models.baseline import Baseline as cls
            return cls(
                model_name=model_name,
                output_file_name=stamped_output_file,
                prompt_generator=prompt_generator,
            )

        if mode == "default":
            if provider == "openai" and model_name in ("gemma-7b", "llama-8b", "llama-70b", "mistral-7b"):
                name_to_cls = {
                    "gemma-7b":   "models.gemma_7b.Gemma7B",
                    "llama-8b":   "models.llama_8b.Llama8b",
                    "llama-70b":  "models.llama_70b.Llama70b",
                    "mistral-7b": "models.mistral_7b.Mistral7B",
                }
                module_path, class_name = name_to_cls[model_name].rsplit(".", 1)
                import importlib
                cls = getattr(importlib.import_module(module_path), class_name)
            else:
                from models.openai_direct_model import OpenAIDirectModel as cls
        elif mode == "plan":
            from models.openai_plan_model import OpenAIPlanModel as cls
        elif mode == "code-plan":
            from models.openai_code_plan_model import OpenAICodePlanModel as cls
        else:
            raise ValueError(f"Unknown mode '{mode}'. Choose from: default, plan, code-plan.")

        return cls(
            model_name=model_name,
            output_file_name=stamped_output_file,
            prompt_generator=prompt_generator,
            provider=provider,
        )
