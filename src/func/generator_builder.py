import logging
from typing import cast

import torch
from transformers import AutoTokenizer, pipeline

from src.func.retriever_builder import RetrieverBuilder

logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# Avoid accelerate's automatic disk/CPU offload, which leaves params on the meta device.
DEVICE_MAP = "auto" if torch.cuda.is_available() else None
DEVICE = None if torch.cuda.is_available() else -1

# Load model and tokenizer at module level for efficiency
logger.info("Loading model: %s", MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
generator = pipeline(
    "text-generation",
    model=MODEL_ID,
    tokenizer=tokenizer,
    device_map=DEVICE_MAP,
    device=DEVICE,
    dtype="auto",
    clean_up_tokenization_spaces=False,  # BPE tokenizers shouldn't strip spaces
)
logger.info("Model loaded successfully")


class Generator:
    """A class for generating answers using retrieved context and a language model."""

    def __init__(self, prompt: str) -> None:
        """Initialize the Generator with a prompt.

        Args:
            prompt: The user query to generate an answer for.
        """
        self.prompt = prompt
        # State attributes
        self.retrieved_text: str = ""
        self.generated_answer: str = ""

    def load_retriever(self) -> str:
        """Retrieve context chunks for the prompt.

        Args:
        Args:
            None.

        Returns:
            Concatenated chunk texts as a single string.
        """
        logger.info("Loading retriever for prompt: %s", self.prompt)
        retriever = RetrieverBuilder(prompt=self.prompt)
        self.retrieved_text = retriever.retrieved_context_str()
        logger.info("Retrieved context of length: %d", len(self.retrieved_text))
        return self.retrieved_text

    def build_prompt(self) -> str:
        """Build the augmented prompt with system message, context, and question.

        Returns:
            The formatted prompt ready for the model.
        """
        if not self.retrieved_text:
            raise ValueError("Retrieved text is empty. Call load_retriever() first.")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides answers based on the provided context.",
            },
            {
                "role": "user",
                "content": f"Context:\n{self.retrieved_text}\n\nQuestion: {self.prompt}",
            },
        ]
        prompt_text = cast(
            str,
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
        )
        logger.info("Prompt built successfully")
        return prompt_text

    def generate_answer(self, prompt_text: str) -> str:
        """Generate an answer using the language model.

        Args:
            prompt_text: The formatted prompt to send to the model.

        Returns:
            The generated answer text.
        """
        logger.info("Generating answer for prompt")
        output = generator(
            prompt_text,
            max_new_tokens=256,  # Explicitly set tokens to generate
        )
        answer = output[0]["generated_text"][len(prompt_text) :]
        logger.info("Answer generated successfully")
        self.generated_answer = answer
        return answer

    def run(self) -> dict[str, str]:
        """Execute the full RAG generation workflow.

        Returns:
            A dictionary containing prompt, retrieved_text, and generated answer.
        """
        logger.info("Starting RAG generation workflow")
        self.load_retriever()
        prompt_text = self.build_prompt()
        answer = self.generate_answer(prompt_text)
        logger.info("RAG generation workflow completed successfully")
        return {
            "prompt": self.prompt,
            "retrieved_text": self.retrieved_text,
            "answer": answer,
        }
