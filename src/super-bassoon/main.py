from llmproxy import LlmProxy
from op import get_secret
from paperless import PaperlessNGX
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

# configuration constants used by example; could be made customizable later
PAPERLESS_URL = "http://192.168.68.222:8000"
PAPERLESS_TOKEN = get_secret("op://homelab/paperless-api-token/credential")

LITELLM_URL = "http://192.168.68.222:4040"
LITELLM_API_KEY = get_secret("op://homelab/litellm-virtual-key-for-claude-code/credential")
EXTRACTOR_MODEL = "openai/claude-gemini-12"
REVIEWER_MODEL = "openai/falcon-7b"
EMBEDDER_MODEL = "openai/nomic-embed-text"

QDRANT_URL = "http://192.168.68.222:6333"
QDRANT_COLLECTION = "receipt_embeddings"


def main():
    ngx = PaperlessNGX(PAPERLESS_URL, PAPERLESS_TOKEN)
    document_types = ngx.get_document_types()
    receipt_id = next((dt["id"] for dt in document_types if dt["name"] == "receipt"), None)
    if receipt_id is None:
        raise RuntimeError("No document type named 'receipt' found in PaperlessNGX")

    receipts = ngx.get_document_ids_by_type(document_type_id=receipt_id)
    if not receipts:
        raise RuntimeError(f"No documents found for document type id {receipt_id}")

    # pick a sample receipt; make sure the index exists
    index = 7
    if index >= len(receipts):
        index = 0
    receipt = ngx.get_document(receipts[index])
    
    print("Extracting structured data from receipt using LiteLLM...")
    llm = LlmProxy(LITELLM_URL, LITELLM_API_KEY)
    try:
        extraction = llm.extract(
            model=EXTRACTOR_MODEL,
            document=receipt,
            document_type="receipt",
        )

        score = llm.review(model=REVIEWER_MODEL, extracted=extraction, document_type="receipt")
        print(f"Review score: {score:.1f}/100")

        summary = llm.summarise(model=EXTRACTOR_MODEL, extracted=extraction, document_type="receipt")
        print(summary)

        print("Generating embedding for receipt summary...")
        vector = llm.embed(model=EMBEDDER_MODEL, text=summary)
        print(f"Embedding vector (first 5 dimensions): {vector[:5]}")

        extraction["summary"] = summary  # add summary to metadata for storage
        client = QdrantClient(QDRANT_URL)

        if not client.collection_exists(QDRANT_COLLECTION):
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )


        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=extraction,  # store the entire extraction + summary as payload
                )
            ],
        )


        # if score >= 80:
        #     print("Generating embedding for receipt data...")
        #     embedding = llm.embed(model=EMBEDDER_MODEL, text=json.dumps(extraction_result))
        #     print(f"Embedding vector (first 5 dimensions): {embedding[:5]}")

        #     # Example of storing the embedding in Qdrant
        #     qdrant = QdrantClient(QDRANT_URL)
        #     embedding = llm.embed(model=EMBEDDER_MODEL, text=json.dumps(extraction_result))

    except Exception as exc:
        print("failed to extract with LiteLLM:", exc)


if __name__ == "__main__":
    value = uuid.uuid5("hello world", str(1234))
    print(str(value))
    main()