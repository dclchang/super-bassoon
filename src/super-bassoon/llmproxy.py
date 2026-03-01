import litellm
import json
import re
from pathlib import Path
from string import Template
from paperless import PaperlessNGX

class LlmProxy:
    def __init__(self, base_url: str, api_key: str, paperless: PaperlessNGX):
        # LiteLLM uses these global variables to direct its traffic
        litellm.api_base = base_url.rstrip("/")
        litellm.api_key = api_key
        
        # Optional: Disable LiteLLM's internal logging for a cleaner console
        litellm.set_verbose = False

        self.paperless = paperless

    def extract(self, model: str, document: dict, document_type: str) -> dict:
        """Extract structured data from a PaperlessNGX document dictionary."""
        system_content = self._load_extraction_prompt(document_type=document_type)

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
    
    def _load_schema(self, document_type: str) -> str:
        schema_dir = Path(__file__).parent / "schemas"
        schema_file = schema_dir / f"{document_type}.txt"
        if not schema_file.exists():
            raise FileNotFoundError(f"Missing schema file: {document_type}")
        return schema_file.read_text()

    def _load_extraction_prompt(self, document_type: str) -> str:
        """Load the prompt template for a given document type."""
        schema = self._load_schema(document_type=document_type)

        prompts_dir = Path(__file__).parent / "prompts"
        prompt_file = prompts_dir / "extraction" / f"{document_type}.txt"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Missing prompt file: {document_type}")
        
        prompt = Template(prompt_file.read_text())
        prompt = prompt.substitute(schema=schema)  # inject the schema into the prompt template
        #return prompt_file.read_text()
        return prompt

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

    def review(self, model: str, extracted: dict, document_type: str) -> dict:
        """Ask the LLM to judge an extraction against the prompt template."""
        template = self._load_extraction_prompt(document_type=document_type)

        system_msg = (
            "You are a reviewer assistant. Compare the provided JSON data to the "
            "description in the prompt template and return a score indicating how well the JSON"
            "objects matches the requirements of the template."
            "Return a valid JSON object that contains the following attributes. DO NOT return any additional text or code fences:"
            "- score: A numeric score from 0 to 100 indicating the quality of the extraction. 100 means perfect match to the template, 0 means completely wrong. Always return a score, even if the JSON is malformed or missing attributes."
            "- issues: A list of any specific issues you found with the extraction (e.g. missing fields, incorrect formats, etc.)"
            "Example:"
            '{"score: 85, "issues": ["field X is missing", "field Y is in the wrong format"]}'
        )

        user_msg = (
            f"Prompt template:\n{template}\n\n"
            f"Extracted JSON:\n{json.dumps(extracted, indent=2)}"
        )
        
        reply = self.chat(model=model, prompt=user_msg, system=system_msg)
        return json.loads(reply.strip())
        # # Parse numeric score
        # m = re.search(r"([0-9]+(?:\.[0-9]+)?)", reply)
        # if not m:
        #     raise ValueError(f"Could not parse score from review reply: {reply!r}")
        
        # score = float(m.group(1))
        # return max(0.0, min(100.0, score))
    
    def vectorise(self, model: str, text: str) -> list[float]:
        """Get an embedding vector for the given text."""
        response = litellm.embedding(model=model, input=[text])
        return response['data'][0]['embedding']

    def query_classifier(self, model: str, query: str, document_types: list[str]) -> str:
        """Classify a user query into a document type (e.g. 'receipt', 'invoice', etc.) based on the content of the query."""
        system_msg = (
            "You are a query classifier assistant."
            "Your task is to analyze the user's query and determine which document type it is referring to."
            f"The possible document types are: {', '.join(document_types)}"
        )

    def query_intent(self, model: str, query: str) -> dict:
        document_type = self.query_classifier(model=model, query=query)
        schema = self._load_schema(document_type=document_type)
        system_msg = (
            f"You are a query intent classifier. "
            f"Your task is to analyze the user's query and determine the intent and document type."
            f"Return a JSON object with the following attributes:"
            f"- document_type: The type of document the query is about (e.g., 'receipt', 'invoice', etc.)"
            f"- intent: The intent of the query (e.g., 'search', 'summarize', 'extract', etc.)"
        )

    def prepare_query(self, model: str, query: str, document_type: str) -> str:
        """Prepare a user query for vector search by generating an embedding."""
        prompt = self._load_extraction_prompt(document_type=document_type)

        system_msg = (
            f"You are a RAG query preparation assistant. "
            "Your task is to convert the user's natural language query into a strucutred search plan."
            "Return the result as a JSON object with the following attributes:"
            f"Use the following prompt template as context:\n{prompt}"
        )
        sys_msg = """
You are a RAG query preparation assistant. Your task is to convert the user's natural language query into a structured search plan. The search plan should be a JSON object with the following attributes:
- query: A concise reformulation of the user's query, optimized for vector search.
- filters: Any relevant metadata filters that should be applied to the search (e.g., vendor, date
"""


        #return self.vectorise(model=model, text=query)
        reply = self.chat(model=model, prompt=query, system=system_msg)
        return reply

