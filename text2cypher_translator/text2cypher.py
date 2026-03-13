import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import logging
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

ADAPTER_NAME = "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"
BASE_MODEL_NAME = "google/gemma-2-9b-it"

INSTRUCTION_TEMPLATE = (
    "Generate Cypher statement to query a graph database. "
    "Use only the provided relationship types and properties in the schema. \n"
    "Schema: {schema} \n Question: {question}  \n Cypher output: "
)


class Text2CypherTranslator:
    def __init__(self, quantize: bool = True):
        self._model, self._tokenizer = self._load_model(quantize)

    @staticmethod
    def _load_model(quantize: bool):
        log.info("Loading base model %s ...", BASE_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_NAME)

        load_kwargs = {
            "dtype": torch.bfloat16,
            "attn_implementation": "eager",
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        }

        if quantize:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME, **load_kwargs)
        log.info("Loading adapter %s ...", ADAPTER_NAME)
        model = PeftModel.from_pretrained(base_model, ADAPTER_NAME)
        log.info("Model loaded.")
        return model, tokenizer

    @staticmethod
    def _build_prompt(question: str, schema: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": INSTRUCTION_TEMPLATE.format(schema=schema, question=question),
            }
        ]

    @staticmethod
    def _postprocess_cypher(raw: str) -> str:
        log.info("Raw output: %s", raw)
        raw, _, _ = raw.partition("**Explanation:**")
        raw = raw.strip("`\n")
        raw = raw.lstrip("cypher\n")
        raw = raw.strip("`\n ")
        return raw

    def generate(self, question: str, schema: str) -> str:
        messages = self._build_prompt(question, schema)
        prompt = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._tokenizer(prompt, return_tensors="pt", padding=True).to(
            self._model.device
        )

        self._model.eval()
        with torch.no_grad():
            tokens = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        output = self._tokenizer.decode(
            tokens[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return self._postprocess_cypher(output)
