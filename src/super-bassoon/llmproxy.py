import asyncio
import json
import re
from pathlib import Path
from string import Template
from typing import Optional, List, Dict, Any

import litellm

from paperless import PaperlessNgx
from op import get_secret


class LlmProxy:
    def __init__(self, base_url: str, api_key: str, models: dict, max_concurrent: int = 1):
        litellm.api_base = base_url.rstrip("/")
        litellm.api_key = api_key

        self.extractor_model = models.get("extractor", "openai/claude-gemini-12")
        self.reviewer_model = models.get("reviewer", "openai/falcon-7b")
        self.embedding_model = models.get("embedding", "openai/nomic-embed-text")

        self._semaphore = asyncio.Semaphore(max_concurrent)

        litellm.set_verbose = False

    async def extract(self, document: dict, document_type: str) -> dict:
        system_content = self._load_extraction_prompt(document_type=document_type)

        prompt_text = document.get("content", "")
        metadata = {k: document[k] for k in ("document_id", "created", "added") if k in document}

        if metadata:
            meta_lines = [f"{k}: {v}" for k, v in metadata.items()]
            prompt_text = "\n".join(meta_lines) + "\n\n" + prompt_text

        raw = await self.chat(
            model=self.extractor_model,
            prompt=prompt_text,
            system=system_content
        )
        return self._parse_response(raw, metadata)

    async def chat(self, model: str, prompt: str, system: str = None, is_json: bool = True) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with self._semaphore:
            response = await litellm.acompletion(
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
        schema = self._load_schema(document_type=document_type)

        prompts_dir = Path(__file__).parent / "prompts"
        prompt_file = prompts_dir / "extraction" / f"{document_type}.txt"

        if not prompt_file.exists():
            raise FileNotFoundError(f"Missing prompt file: {document_type}")

        prompt = Template(prompt_file.read_text())
        prompt = prompt.substitute(schema=schema)
        return prompt

    def _parse_response(self, raw: str, metadata: dict) -> dict:
        try:
            result = json.loads(raw.strip())
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    result = json.loads(m.group(0))
                except:
                    return raw
            else:
                return raw

        if isinstance(result, dict) and metadata:
            for k, v in metadata.items():
                result.setdefault(k, v)
        return result

    async def summarise(self, extracted: dict, document_type: str) -> str:
        system_msg = (
            f"You are a data processing assistant. Your task is to convert structered JSON "
            "that represents a {document_type} into a single, concise, natural language paragraph."
            "Do not return anu markdown (no bolding, headers or bullet points)."
            "Be concise and always start with 'Receipt from [Vendor]...', no other start of the sentence is acceptable."
        ).format(document_type=document_type)

        user_msg = f"Extracted JSON:\n{json.dumps(extracted, indent=2)}\n\n"

        summary = await self.chat(
            model=self.extractor_model,
            prompt=user_msg,
            system=system_msg,
            is_json=False
        )
        return summary.strip()

    async def review(self, extracted: dict, document_type: str) -> dict:
        template = self._load_extraction_prompt(document_type=document_type)

        system_msg = (
            "You are a reviewer assistant. Compare the provided JSON data to the "
            "description in the prompt template and return a score indicating how well the JSON"
            "objects matches the requirements of the template."
            "Return a valid JSON object that contains the following attributes. DO NOT return any additional text or code fences:"
            "- score: A numeric score from 0 to 100 indicating the quality of the extraction. 100 means perfect match to the template, 0 means completely wrong. Always return a score, even if the JSON is malformed or missing attributes."
            "- issues: A list of any specific issues you found with the extraction (e.g. missing fields, incorrect formats, etc.)"
            "Example:"
            '{"score": 85, "issues": ["field X is missing", "field Y is in the wrong format"]}'
        )

        user_msg = (
            f"Prompt template:\n{template}\n\n"
            f"Extracted JSON:\n{json.dumps(extracted, indent=2)}"
        )

        reply = await self.chat(
            model=self.reviewer_model,
            prompt=user_msg,
            system=system_msg
        )
        return json.loads(reply.strip())

    async def vectorise(self, text: str) -> List[float]:
        async with self._semaphore:
            response = await litellm.aembedding(
                model=self.embedding_model,
                input=[text]
            )
        return response['data'][0]['embedding']

    async def query_classifier(self, query: str, document_types: List[str]) -> str:
        system_msg = (
            "You are a query classifier assistant."
            "Your task is to analyze the user's query and determine which document type it is referring to."
            f"The possible document types are: {', '.join(document_types)}"
            "Return only ONE best matching answer and nothing else, no explanations, no markdown, no code fences, just the document type as a single word in lowercase."
        )
        reply = await self.chat(
            model=self.extractor_model,
            prompt=query,
            system=system_msg,
            is_json=False
        )
        return reply.strip().lower()

    async def query_filters(self, query: str, document_type: str) -> dict:
        schema = self._load_schema(document_type=document_type)
        system_msg = Template('''
You are a RAG query planning assistant. Given a natural language question, you must return a JSON object 
that represents a Qdrant filter. The JSON must conform exactly to the following structure:

{
    "must": [                          # ALL conditions must match (AND)
        {
            "key": "<field_name>",
            "match": {"value": <str or int or bool>}   # exact match
        },
        {
            "key": "<field_name>",
            "range": {                 # for numeric or date comparisons
                "gt":  <number>,       # greater than (optional)
                "gte": <number>,       # greater than or equal (optional)
                "lt":  <number>,       # less than (optional)
                "lte": <number>        # less than or equal (optional)
            }
        }
    ],
    "should": [...],                   # ANY condition must match (OR)
    "must_not": [...]                  # NOT conditions
}

Rules:
- Only include "must", "should", or "must_not" keys if they are needed
- Only use field names from this list: [vendor, document_type, total_amount, date, correspondent]
- For dates use ISO 8601 format: "2024-01-31"
- Return ONLY the JSON object, no explanation or markdown

Example input: "find electricity bills from AGL under $$100 in 2024"
Example output:
{
    "must": [
        {"key": "document_type", "match": {"value": "bill"}},
        {"key": "vendor", "match": {"value": "AGL"}},
        {"key": "total_amount", "range": {"lt": 100}},
        {"key": "date", "range": {"gte": "2024-01-01", "lte": "2024-12-31"}}
    ]
}

Refer to the following schema for field definitions:
$schema
''')
        system_msg = system_msg.substitute(schema=schema)
        reply = await self.chat(
            model=self.extractor_model,
            prompt=query,
            system=system_msg
        )
        return json.loads(reply)


if __name__ == "__main__":
    import asyncio

    async def main():
        paperless = PaperlessNgx(
            base_url="http://192.168.68.222:8000",
            api_key=get_secret("op://homelab/paperless-api-token/credential"))
        llmproxy = LlmProxy(
            base_url="http://192.168.68.222:4040",
            api_key=get_secret("op://homelab/litellm-virtual-key-for-rag-app/credential"),
            models={
                "extractor": "openai/claude-gemini-12",
                "reviewer": "openai/falcon-7b",
                "embedding": "openai/nomic-embed-text"
            })
        classification = await llmproxy.query_filters(
            query="When was my Aussie Broadband under $100?",
            document_type="receipt"
        )
        print(classification)
        await paperless.close()

    asyncio.run(main())
