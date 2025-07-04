from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
from langchain_huggingface.llms import HuggingFacePipeline

class LLMModel:
    def __init__(self, model_name: str, quantization = True):

        nf4_config = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_compute_dtype=torch.bfloat16
                                    )
        self.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config= nf4_config if quantization == True else None,
        low_cpu_mem_usage=True
    )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model_pipeline = pipeline(
                                "text-generation",
                                model=self.model,
                                tokenizer=self.tokenizer,
                                max_new_tokens=512,
                                pad_token_id=self.tokenizer.eos_token_id,
                                device_map="auto"
                            )
        self.llm = HuggingFacePipeline(pipeline=self.model_pipeline)


    def generate(self, prompt: str):
        return self.llm(prompt)