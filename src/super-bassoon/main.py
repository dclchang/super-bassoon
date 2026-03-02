import asyncio
import json
import uuid

from llmproxy import LlmProxy
from op import get_secret
from paperless import PaperlessNgx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from vectordb import VectorDb

PAPERLESS_URL = "http://192.168.68.222:8000"
PAPERLESS_TOKEN = get_secret("op://homelab/paperless-api-token/credential")

LITELLM_URL = "http://192.168.68.222:4040"
LITELLM_API_KEY = get_secret("op://homelab/litellm-virtual-key-for-rag-app/credential")
EXTRACTOR_MODEL = "openai/claude-gemini-12"
REVIEWER_MODEL = "openai/falcon-7b"
EMBEDDER_MODEL = "openai/nomic-embed-text"

QDRANT_URL = "http://192.168.68.222:6333"
QDRANT_COLLECTION = "receipt_embeddings"


async def main():
    ngx = PaperlessNgx(PAPERLESS_URL, PAPERLESS_TOKEN)

    receipts = await ngx.get_document_ids_by_type(document_type='receipt')
    if not receipts:
        raise RuntimeError(f"No documents found for document type 'receipt'")

    index = 7
    if index >= len(receipts):
        index = 0
    receipt = await ngx.get_document(receipts[index])

    print("Extracting structured data from receipt using LiteLLM...")
    llm = LlmProxy(LITELLM_URL, LITELLM_API_KEY, {
        "extractor": EXTRACTOR_MODEL,
        "reviewer": REVIEWER_MODEL,
        "embedding": EMBEDDER_MODEL,
    })

    try:
        extraction = await llm.extract(
            document=receipt,
            document_type="receipt",
        )

        score = await llm.review(extracted=extraction, document_type="receipt")
        print(f"Review score: {score:.1f}/100")

        summary = await llm.summarise(extracted=extraction, document_type="receipt")
        print(summary)

        print("Generating embedding for receipt summary...")
        vector = await llm.vectorise(text=summary)
        print(f"Embedding vector (first 5 dimensions): {vector[:5]}")

        extraction["summary"] = summary

        vectordb = VectorDb(base_url=QDRANT_URL)
        vectordb.upsert(vector=vector, payload=extraction, collection_name=QDRANT_COLLECTION)

    except Exception as exc:
        print("failed to extract with LiteLLM:", exc)
    finally:
        await ngx.close()


if __name__ == "__main__":
    asyncio.run(main())
