"""
Inference with the fine-tuned CodeT5+ text2cypher model.

Drop-in replacement for the Gemma-based Text2CypherTranslator.
"""

import logging

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "./codet5_text2cypher"

INSTRUCTION_PREFIX = (
    "Generate Cypher statement to query a graph database.\n"
    "Schema: {schema}\nQuestion: {question}\nCypher: "
)


class Text2CypherTranslatorCodeT5:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        log.info("Loading CodeT5+ model from %s ...", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = T5ForConditionalGeneration.from_pretrained(model_path)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)
        self._model.eval()
        log.info("Model loaded on %s", self._device)

    def generate(self, question: str, schema: str) -> str:
        prompt = INSTRUCTION_PREFIX.format(schema=schema, question=question)
        inputs = self._tokenizer(
            prompt, return_tensors="pt", max_length=512, truncation=True
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
            )

        cypher = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return cypher.strip()
