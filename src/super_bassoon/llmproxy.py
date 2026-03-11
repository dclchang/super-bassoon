import asyncio
import datetime
import json
import re
from pathlib import Path
from qdrant_client.models import ScoredPoint, Filter
from string import Template
from typing import Dict, List, Optional, Union, cast

import litellm

from super_bassoon.paperless import PaperlessNgx
from super_bassoon.op import get_secret


class LlmProxy:
    def __init__(self, base_url: str, api_key: str, models: dict, max_concurrent: int = 1):
        litellm.api_base = base_url.rstrip("/")
        litellm.api_key = api_key

        self.extractor_model = models.get("extractor", "openai/claude-gemini-12")
        self.reviewer_model = models.get("reviewer", "openai/falcon-7b")
        self.embedding_model = models.get("embedding", "openai/nomic-embed-text")
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def extract(self, document: dict, document_type: str) -> dict:
        system_content = self._load_extraction_prompt(document_type=document_type)

        prompt_text = document.get("content", "")
        metadata = {k: document[k] for k in ("document_id", "created", "added") if k in document}
        metadata["document_type"] = document_type

        if metadata:
            meta_lines = [f"{k}: {v}" for k, v in metadata.items()]
            prompt_text = "\n".join(meta_lines) + "\n\n" + prompt_text

        raw = await self.chat(
            model=self.extractor_model,
            prompt=prompt_text,
            system=system_content
        )
        return self._parse_response(raw, metadata)

    def chatsync(self, model: str, prompt: str, system: str, is_json: bool = True) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        
        response = cast(litellm.ModelResponse, litellm.completion(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"} if is_json else None
        ))
        choice = response.choices[0] if response.choices else None  # type: ignore[union-attr]
        content = choice.message.content if choice and choice.message else None  # type: ignore[union-attr]

        if content is None:
            raise ValueError("Empty response from LLM")
        return content


    async def chat(self, model: str, prompt: str, system: str, is_json: bool = False, json_schema: Optional[Dict] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response_format: Union[Dict, None] = None
        if is_json:
            response_format = {"type": "json_object"}
            if json_schema:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "user_schema",
                        "schema": json_schema
                    }
                }

        async with self._semaphore:
            response = cast(litellm.ModelResponse, await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=0,
                response_format=response_format
            ))
            
        choice = response.choices[0] if response.choices else None  # type: ignore[union-attr]
        content = choice.message.content if choice and choice.message else None  # type: ignore[union-attr]

        if content is None:
            raise ValueError("Empty response from LLM")
        return content

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
                    raise ValueError(f"Failed to parse JSON from LLM response: {raw[:100]}...")
            else:
                raise ValueError(f"No JSON object found in LLM response: {raw[:100]}...")

        if not isinstance(result, dict):
            raise ValueError(f"LLM response is not a dict: {type(result).__name__}")

        if metadata:
            for k, v in metadata.items():
                result.setdefault(k, v)
        return result

    async def generate_questions(self, summary: str) -> list[str]:
        system_msg = """
You are a helpful assistant that generates questions based on a receipt summary.

Given a receipt summary, generate 5 different questions that a person might ask 
when trying to find THIS specific receipt from their personal expense history.

Each question must:
- Be specific enough that only this receipt would be a good answer
- Reference the specific vendor, service type, brand, or product where relevant
- Cover a MIX of specific recall ("STIHL brushcutter at Five Ways") and 
  broader recall ("outdoor power tools", "garden machinery") — not all five 
  questions should reference the vendor name
- Reflect natural conversational language a person uses when recalling a purchase
  e.g. "my internet bill", "Aussie Broadband payment", "monthly broadband cost"
- Where dates are involved, use specific dates or time frames from the summary
  e.g. "in November 2024" rather than "last month" or "last October" or "recently"

Avoid:
- Generic questions like "how much was spent?" or "what company was paid?" 
  that could apply to ANY receipt
- Relative time references like "last October" or "recently" 
  — use specific dates from the summary instead
- Retrieval instructions like "can you find..." or "show me..." 
  — phrase as direct questions instead

IF the summary does not contain enough specific information to generate 5 unique questions,
then generate as fewer but higher quality questions. Aim for a minimum of 2 and a maximum of 5.


Return ONLY a JSON array of 5 strings, no additional text, explanation or markdown.
"""
        response = await self.chat(
            model=self.extractor_model,
            prompt=summary,
            system=system_msg
        )
        return json.loads(response)

    async def summarise(self, extracted: dict, document_type: str) -> str:
            system_msg = """
    You are a summarisation assistant. Your task is to take structured JSON data that has been extracted 
    from a document and write a 3-4 sentence plain English description.

    Rules:
    - mention the vendor and any brand names of products purchased
    - describe what the items actually ARE, not just their names
    (e.g. "Philips Hue" → "Philips Hue smart lighting / smart bulb")
    - include the category of purchase in natural language
    - mention the total amount and date naturally
    - include implicit context a person would understand
    (e.g. "JB Hi-Fi" → "electronics retailer", "Nando's" → "restaurant / dining out")

    Example output:
    "Spent $89.95 at JB Hi-Fi, an electronics retailer, on 3 March 2024. 
    Purchased a Philips Hue smart bulb starter kit — a smart home lighting 
    product by Philips. This was a electronics purchase in the smart home 
    sub-category."
    """.format(document_type=document_type)

            user_msg = f"Extracted JSON:\n{json.dumps(extracted, indent=2)}\n\n"

            summary = await self.chat(
                model=self.extractor_model,
                prompt=user_msg,
                system=system_msg
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

    async def query_classifier(self, query: str, document_types: List[str]) -> List[str]:
        system_msg = (
            "You are a query classifier assistant."
            "Your task is to analyze the user's query and determine all document types that could potentially match the users query"
            "It is better to return more possible matches than miss a relevant or even related document type."
            "If the query is very general and could apply to any document type, return all document types."
            f"The possible document types are: {', '.join(document_types)}"
            "Return a JSON array of ALL matching document types in lowercase. Example: [\"receipt\", \"bill\"]. No explanations, no markdown."
        )
        reply = await self.chat(
            model=self.extractor_model,
            prompt=query,
            system=system_msg,
            is_json=True
        )
        try:
            result = json.loads(reply.strip())
            if isinstance(result, list):
                return result
            return [result]
        except json.JSONDecodeError:
            return [reply.strip().lower()]

    async def get_top_k(self, query: str, document_type: str) -> int:
        system_msg = (
            "You are a query analysis assistant. "
            "Your task is to determine the appropriate number of results (k) to return for a vector database query based on the user's natural language query."
            f"Consider the specificity of the user's query for the given document type {document_type} and how many relevant results are likely needed to satisfy it."
            "Return only a single integer number (e.g. 5) with no additional text, explanations, or formatting."
        )
        reply = await self.chat(
            model=self.extractor_model,
            prompt=query,
            system=system_msg,
            is_json=False
        )
        try:
            k = int(reply.strip())
            return max(1, min(k, 20))  # constrain k to be between 1 and 20
        except ValueError:
            return 5  # default value if parsing fails


    async def answer_question(self, question: str, scored_points: list) -> str:
        ANSWER_PROMPT = """
        You are a personal finance assistant. Answer the user's question using ONLY 
        the receipts provided below.

        Rules:
        - Only use information explicitly present in the receipts
        - If the receipts don't contain enough information to answer, say so clearly
        - Do not guess, infer, or use outside knowledge
        - If asked for a total or average, calculate it from the receipt amounts provided
        - Cite which receipt(s) you're drawing from in your answer

        RECEIPTS:
        {context}

        USER QUESTION:
        {question}
        """
        deduped = self.deduplicate(scored_points)
        context = "\n\n".join([
            f"Receipt {i+1}:\n{json.dumps(point.payload, indent=2)}"
            for i, point in enumerate(deduped)
        ])
        
        response = litellm.completion(
            #model="gemini/gemini-2.5-flash",
            model=self.extractor_model,
            messages=[
                {"role": "system", "content": ANSWER_PROMPT.format(
                    context=context,
                    question=question
                )}
            ]
        )
        return response.choices[0].message.content  # type: ignore

    def deduplicate(self, scored_points: list) -> list:
        seen = {}
        for point in scored_points:
            source_id = point.payload["document_id"]
            if source_id not in seen or point.score > seen[source_id].score:
                seen[source_id] = point  # keep highest scoring point per receipt
        return list(seen.values())

    async def get_filters(self, query: str, document_types: List[str]) -> Filter:
        document_type = document_types[0]
        #schema = self._load_schema(document_type=document_type)
        today = datetime.date.today().isoformat()
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

## document_type
You MUST include a document_type filter using the "any" clause to match ALL of the following document types: $document_types.
For example, if document_types is ["invoice", "receipt"], you must return:
{"key": "document_type", "match": {"any": ["invoice", "receipt"]}}

## total_amount
Only extract if the user explicitly mentions a dollar amount or price.
Apply a range based on the user's wording:
- No qualifier ("I spent $$300")      → gte: 297.00, lte: 303.00  (±1%)
- "about / around / roughly" $$300    → gte: 255.00, lte: 345.00  (±15%)
- "approximately" $$300               → gte: 240.00, lte: 360.00  (±20%)
- "over / more than" $$300            → gte: 300.00  (no upper bound)
- "under / less than" $$300           → lte: 300.00  (no lower bound)
- "between $$200 and $$400"            → gte: 200.00, lte: 400.00
Place total_amount in "should", never "must".
DO NOT include total_amount in the filter unless the user's message contains 
an explicit dollar figure or price (e.g. "$$300", "300 dollars", "three hundred dollars").
If the user asks "how much did I pay" or "what did it cost" — this is a QUESTION 
about an amount, NOT a filter on an amount. Omit total_amount entirely.
If no explicit amount is stated, omit total_amount entirely. No exceptions.

## purchase_date
Only extract if the user mentions a specific date, month, year, or time period.
Convert relative terms based on today's date ($today).
- "last month"                       → gte: first day of last month, lte: last day of last month
- "last year"                        → gte: 2024-01-01, lte: 2024-12-31
- "in February"                      → gte: 2025-02-01, lte: 2025-02-28
- "recently" / "the other day"       → omit, too vague to filter
- "before March"                     → lte: 2025-02-28  (no lower bound)
- "since January"                    → gte: 2025-01-01  (no upper bound)
Place purchase_date in "must".
DO NOT include purchase_date in the filter unless the user's message contains 
an explicit date (e.g. "Nov 2024", "last year").
If the user asks "when did I go" or "when was" — this is a QUESTION 
about an amount, NOT a filter on a date. Omit purchase_date entirely.
If no explicit time frame is stated, omit purchase_date entirely. No exceptions.

                              
Example input: "find electricity bills from AGL under $$100 in 2024"
Example output:
{
    "must": [
        {"key": "total_amount", "range": {"lt": 100}},
        {"key": "purchase_date", "range": {"gte": "2024-01-01", "lte": "2024-12-31"}}
    ]
}
''')
        system_msg = system_msg.substitute(today=today, document_types=document_types)

        filter_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "must": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "key": {"const": "document_type"},
                            "match": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "any": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                },
                                "required": ["any"]
                            }
                        },
                        "required": ["key", "match"]
                    }
                },
                "should": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "key": {"const": "purchase_date"},
                                    "range": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "minProperties": 1,
                                        "properties": {
                                            "gt":  {"type": ["number", "string"]},
                                            "gte": {"type": ["number", "string"]},
                                            "lt":  {"type": ["number", "string"]},
                                            "lte": {"type": ["number", "string"]}
                                        }
                                    }
                                },
                                "required": ["key", "range"]
                            },
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "key": {"const": "total_amount"},
                                    "range": {
                                        "type": "object",
                                        "additionalProperties": False,
                                        "minProperties": 1,
                                        "properties": {
                                            "gt":  {"type": "number"},
                                            "gte": {"type": "number"},
                                            "lt":  {"type": "number"},
                                            "lte": {"type": "number"}
                                        }
                                    }
                                },
                                "required": ["key", "range"]
                            }
                        ]
                    }
                },
                "must_not": {"type": "array"}
            },
            "required": ["must"]
        }

        reply = await self.chat(
            model=self.extractor_model,
            prompt=query,
            system=system_msg,
            json_schema=filter_schema
        )
        filter_result = json.loads(reply)
        
        return filter_result

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
    classification = await llmproxy.get_filters(
        query="When was my Aussie Broadband under $100?",
        document_types=["receipt"]
    )
    print(classification)
    await paperless.close()

if __name__ == "__main__":
    #asyncio.run(main())
    llmproxy = LlmProxy(
        base_url="http://192.168.68.222:4040",
        api_key=get_secret("op://homelab/litellm-virtual-key-for-rag-app/credential"),
        models={
            #"extractor": "openai/claude-gemini-12",
            #"extractor": "gemini/gemini/gemini-2.5-flash",
            #"extractor": "openai/qwen3",
            #"extractor": "openai/nous-hermes-2-pro",
            "extractor": "openai/qwen3",
            "reviewer": "openai/falcon-7b",
            "embedding": "openai/nomic-embed-text"
        })
    response = llmproxy.chatsync(
        model=llmproxy.extractor_model,
        prompt="How are you?",
        system=(
            "You are a helpful assistant that answers questions truthfully and informatively. "
        ))
    print(response)

