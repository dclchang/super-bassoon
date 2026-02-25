import requests
from pathlib import Path
import json
import re
#from typing import Union


class LiteLLM:
    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def extract(self, document: dict, document_type: str) -> dict:
        """Extract structured data from a PaperlessNGX document dictionary.

        The caller must supply the entire dictionary returned by
        :meth:`PaperlessNGX.get_document`.  ``content`` is used as the text
        prompt, and all other keys are sent as metadata so the model can echo
        them back (e.g. ``document_id``).

        :param document: dictionary representing a PaperlessNGX document.  This
            value is assumed to always be a dict; passing anything else is a
            programmer error.
        :param document_type: name of the file under ``prompts/`` to load.
        """
        prompts_dir = Path(__file__).parent / "prompts"
        prompt_file = next(prompts_dir.glob(f"{document_type}.txt"), None)
        if prompt_file is None:
            raise FileNotFoundError(f"No prompt file found for document type: {document_type}")
        system = prompt_file.read_text()

        # caller guarantees dict; use content field and send only the
        # three relevant metadata fields (id/created/added) to the model.
        prompt_text = document.get("content", "")
        metadata = {k: document[k] for k in ("document_id", "created", "added")
                    if k in document}

        if metadata:
            meta_lines = [f"{k}: {v}" for k, v in metadata.items()]
            prompt_text = "\n".join(meta_lines) + "\n\n" + prompt_text

        raw = self.chat(prompt=prompt_text, system=system)
        return self._parse_response(raw, metadata)

    def _parse_response(self, raw: str, metadata: dict) -> dict:
        """Clean up a raw model reply and attempt to return a dict.

        - strip code fences and surrounding whitespace
        - locate a JSON object if the model added extra prose
        - ``json.loads`` the result, falling back to the cleaned string on
          failure
        - merge metadata keys back into the dict if they were omitted by the
          model
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[-1]
            cleaned = cleaned.rsplit("```")[-1].strip()

        result = None
        try:
            result = json.loads(cleaned)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group(0))
                except Exception:
                    result = None
        if result is None:
            return cleaned

        if metadata and isinstance(result, dict):
            for k, v in metadata.items():
                result.setdefault(k, v)
        return result

    def chat(self, prompt: str, system: str = None) -> str:
        """Send a prompt and return the assistant's reply as a string."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={"model": self.model, "messages": messages},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def review(self, extracted: dict, document_type: str) -> float:
        """Ask the LLM to judge an extraction against the prompt template.

        ``extracted`` is the JSON object returned by :meth:`extract`.  The
        method looks up the corresponding ``.txt`` file in ``prompts/`` using
        ``document_type`` and sends both the template and the JSON to the LLM
        with a system prompt that instructs it to validate the match.

        The assistant is expected to return a short analysis that *includes* a
        numeric score between 0 and 100.  This method will attempt to parse
        that score and return it as a float; if parsing fails the method will
        raise a ``ValueError`` so the caller knows something unexpected happened.
        """
        prompts_dir = Path(__file__).parent / "prompts"
        prompt_file = next(prompts_dir.glob(f"{document_type}.txt"), None)
        if prompt_file is None:
            raise FileNotFoundError(f"No prompt file found for document type: {document_type}")
        template = prompt_file.read_text()

        system_msg = (
            "You are a reviewer assistant.  Compare the provided JSON data to the "
            "description in the prompt template and determine if the values are "
            "appropriate and complete.  Respond with a short analysis and a score "
            "from 0 to 100 (higher is better). Do not add any extra prose outside "
            "the analysis."
        )

        user_msg = (
            "Prompt template:\n" + template + "\n\n" +
            "Extracted JSON:\n" + json.dumps(extracted, indent=2)
        )
        reply = self.chat(prompt=user_msg, system=system_msg)

        # attempt to pull a numeric score out of the model's text
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)", reply)
        if not m:
            raise ValueError(f"Could not parse score from review reply: {reply!r}")
        score = float(m.group(1))
        # clamp between 0 and 100
        score = max(0.0, min(100.0, score))
        return score
