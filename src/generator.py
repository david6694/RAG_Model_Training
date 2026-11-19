import json
import os
from typing import Any, List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm

from src.chunker import Chunk
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class AnswerGenerator:
    """LLM Answer Generator using Qwen model with fallback support."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: Optional[str] = None,
    ) -> None:
        """Initialize generator device and lazy load model parameters."""
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer: Any = None
        self.model: Any = None
        self.load_attempted = False

    def _load_model(self) -> bool:
        """Attempt to load LLM model and tokenizer from HuggingFace."""
        if self.load_attempted:
            return self.model is not None

        self.load_attempted = True
        try:
            print(f"Loading LLM model '{self.model_name}' on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            model_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=model_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
            )
            if self.device == "cpu" and self.model is not None:
                self.model.to("cpu")
            print("LLM model successfully loaded.")
            return True
        except Exception as e:
            print(f"Warning: Failed to load LLM model ({e}). Using fallback.")
            self.tokenizer = None
            self.model = None
            return False

    def generate_answer(
        self,
        question: str,
        sources: List[MinimalSource],
        chunks_map: Optional[dict[str, Chunk]] = None,
    ) -> str:
        """Generate grounded answer from question and source attributions.

        Falls back to extractive summary if model loading or generation fails.
        """
        context_snippets: List[str] = []
        if chunks_map:
            for s in sources:
                key = (
                    f"{s.file_path}:"
                    f"{s.first_character_index}:"
                    f"{s.last_character_index}"
                )
                if key in chunks_map:
                    context_snippets.append(chunks_map[key].text)

        context_text = (
            "\n\n".join(context_snippets[:5])
            if context_snippets
            else "No specific source context provided."
        )

        # Truncate context to max 3000 chars to respect prompt budget
        if len(context_text) > 3000:
            context_text = context_text[:3000] + "..."

        if not self._load_model() or self.tokenizer is None or self.model is None:
            return self._fallback_answer(question, context_text)

        try:
            system_prompt = (
                "You are an expert technical assistant specialized in the "
                "vLLM codebase and documentation.\n"
                "Answer the user's question directly, accurately, and coherently "
                "based strictly on the provided context.\n"
                "Ensure your answer is self-contained, well-structured, and "
                "easy to understand."
            )
            user_content = (
                f"Context information:\n{context_text}\n\n"
                f"Question: {question}"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            has_chat_template = (
                hasattr(self.tokenizer, "apply_chat_template")
                and self.tokenizer.chat_template
            )
            if has_chat_template:
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            else:
                prompt = (
                    f"{system_prompt}\n\n"
                    f"Context information:\n{context_text}\n\n"
                    f"Question: {question}\n\n"
                    "Answer:"
                )

            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_len:]
            raw_answer = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            # Strip reasoning tags if model outputted them
            answer = str(raw_answer).strip()
            if "</think>" in answer:
                answer = answer.split("</think>")[-1].strip()
            elif "<think>" in answer:
                answer = answer.replace("<think>", "").strip()

            if not answer:
                return self._fallback_answer(question, context_text)
            return answer
        except Exception as e:
            print(f"Warning: LLM generation error ({e}). Using fallback.")
            return self._fallback_answer(question, context_text)

    def _fallback_answer(self, question: str, context: str) -> str:
        """Generate deterministic extractive answer when LLM is unavailable."""
        if not context or context == "No specific source context provided.":
            return f"Information for '{question}' not found in context."

        first_line = context.splitlines()[0] if context.splitlines() else context
        return f"Based on retrieved sources: {first_line[:200]}"

    def answer_search_results(
        self,
        search_res: MinimalSearchResults,
        chunks_map: Optional[dict[str, Chunk]] = None,
    ) -> MinimalAnswer:
        """Convert MinimalSearchResults into MinimalAnswer with generated answer."""
        answer_str = self.generate_answer(
            question=search_res.question,
            sources=search_res.retrieved_sources,
            chunks_map=chunks_map,
        )
        return MinimalAnswer(
            question_id=search_res.question_id,
            question=search_res.question,
            retrieved_sources=search_res.retrieved_sources,
            answer=answer_str,
        )

    def batch_answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        chunks_map: Optional[dict[str, Chunk]] = None,
    ) -> StudentSearchResultsAndAnswer:
        """Batch answer all search results in a JSON file."""
        filename = (
            os.path.basename(student_search_results_path)
            if student_search_results_path
            else "student_search_results_and_answer.json"
        )
        if not os.path.exists(student_search_results_path):
            print(f"Warning: File '{student_search_results_path}' not found.")
            empty_out = StudentSearchResultsAndAnswer(search_results=[], k=10)
            self._save_answered_results(
                empty_out, save_directory, filename=filename
            )
            return empty_out

        try:
            with open(student_search_results_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            student_res = StudentSearchResults.model_validate(s_data)
        except Exception as e:
            print(f"Error loading search results '{student_search_results_path}': {e}")
            empty_out = StudentSearchResultsAndAnswer(search_results=[], k=10)
            self._save_answered_results(
                empty_out, save_directory, filename=filename
            )
            return empty_out

        answered_list: List[MinimalAnswer] = []
        for s_res in tqdm(student_res.search_results, desc="Generating answers"):
            ans = self.answer_search_results(s_res, chunks_map=chunks_map)
            answered_list.append(ans)

        final_out = StudentSearchResultsAndAnswer(
            search_results=answered_list, k=student_res.k
        )
        self._save_answered_results(
            final_out, save_directory, filename=filename
        )
        return final_out

    def _save_answered_results(
        self,
        res: StudentSearchResultsAndAnswer,
        save_directory: str,
        filename: str = "student_search_results_and_answer.json",
    ) -> str:
        """Save results to JSON file."""
        if save_directory.endswith(".json"):
            out_file = save_directory
            dir_name = os.path.dirname(out_file)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
        else:
            os.makedirs(save_directory, exist_ok=True)
            out_file = os.path.join(save_directory, filename)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(res.model_dump_json(indent=2))
        print(f"Saved answered results to '{out_file}'.")
        return out_file
