import litellm
from pathlib import Path
import json
import re

class LlmProxy:
    def __init__(self, base_url: str, api_key: str = ""):
        # LiteLLM uses these global variables to direct its traffic
        litellm.api_base = base_url.rstrip("/")
        litellm.api_key = api_key
        
        # Optional: Disable LiteLLM's internal logging for a cleaner console
        litellm.set_verbose = False 

    def extract(self, model: str, document: dict, document_type: str) -> dict:
        """Extract structured data from a PaperlessNGX document dictionary."""
        prompts_dir = Path(__file__).parent / "prompts"
        prompt_file = prompts_dir / f"{document_type}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"No prompt file found for: {document_type}")
        
        system_content = prompt_file.read_text()

        # Build user prompt with metadata
        prompt_text = document.get("content", "")
        metadata = {k: document[k] for k in ("document_id", "created", "added") if k in document}

        if metadata:
            meta_lines = [f"{k}: {v}" for k, v in metadata.items()]
            prompt_text = "\n".join(meta_lines) + "\n\n" + prompt_text

        # Call the refactored chat method
        raw = self.chat(model=model, prompt=prompt_text, system=system_content, is_json=True)
        return self._parse_response(raw, metadata)

    def chat(self, model: str, prompt: str, system: str = None, is_json: bool = False) -> str:
        """Send a prompt using the LiteLLM SDK."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # LiteLLM's universal completion function
        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"} if is_json else None
        )
        
        return response.choices[0].message.content

    def _parse_response(self, raw: str, metadata: dict) -> dict:
        """
        Clean up model reply. Since we use json_object mode, 
        cleaning is usually unnecessary, but we keep a fallback.
        """
        try:
            result = json.loads(raw.strip())
        except json.JSONDecodeError:
            # Fallback to regex if the model somehow ignores the JSON format constraint
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group(0))
                except:
                    return raw
            else:
                return raw

        # Re-inject metadata if missing
        if isinstance(result, dict) and metadata:
            for k, v in metadata.items():
                result.setdefault(k, v)
        return result
    
    def summarise(self, model: str, extracted: dict, document_type: str) -> str:
        """Generate a concise summary of the extracted data."""
        system_msg = (
            f"You are a data processing assistant. Your task is to convert structered JSON that represents a {document_type} into a single, concise, natural language paragraph."
            "Do not return anu markdown (no bolding, headers or bullet points)."
            "Be concise and always start with 'Receipt from [Vendor]...', no other start of the sentence is acceptable."
        )

        user_msg = (
            f"Extracted JSON:\n{json.dumps(extracted, indent=2)}\n\n"
        )

        summary = self.chat(model=model, prompt=user_msg, system=system_msg)
        return summary.strip()

    def review(self, model: str, extracted: dict, document_type: str) -> float:
        """Ask the LLM to judge an extraction against the prompt template."""
        prompts_dir = Path(__file__).parent / "prompts"
        prompt_file = prompts_dir / f"{document_type}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Missing prompt file: {document_type}")
            
        template = prompt_file.read_text()

        system_msg = (
            "You are a reviewer assistant. Compare the provided JSON data to the "
            "description in the prompt template. Respond with only a number from 0 to 100. "
            "Do not return anything except the numeric score."
        )

        user_msg = (
            f"Prompt template:\n{template}\n\n"
            f"Extracted JSON:\n{json.dumps(extracted, indent=2)}"
        )
        
        reply = self.chat(model=model, prompt=user_msg, system=system_msg)

        # Parse numeric score
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", reply)
        if not m:
            raise ValueError(f"Could not parse score from review reply: {reply!r}")
        
        score = float(m.group(1))
        return max(0.0, min(100.0, score))
    
    def vectorise(self, model: str, text: str) -> list[float]:
        """Get an embedding vector for the given text."""
        response = litellm.embedding(model=model, input=[text])
        return response['data'][0]['embedding']
