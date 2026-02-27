from llmproxy import LlmProxy
from vectordb import VectorDb
from qdrant_client import QdrantClient
from op import get_secret

if __name__ == "__main__":
    llm = LlmProxy(base_url="http://192.168.68.222:4040",
                   api_key=get_secret("op://homelab/litellm-virtual-key-for-claude-code/credential"))
    vector = llm.vectorise(model="openai/nomic-embed-text", text="How much did I pay VicRoads?")

    client = QdrantClient(url="http://192.168.68.222:6333")
    results = client.query_points(
        collection_name="receipt_collection",
        query=vector,
        limit=5
    )

    for hit in results.points:
        print(f"Score: {hit.score:.4f}")
        print(f"Content: {hit.payload.get('text')}")
