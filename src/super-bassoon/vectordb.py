from op import get_secret
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

class VectorDb:
    def __init__(self, base_url: str):
        self.client = QdrantClient(url=base_url)
        self.namespace = uuid.UUID(get_secret("op://homelab/qdrant-namespace/credential"))

    def _check_collection(self, collection_name: str):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def upsert(self, vector: list[float], payload: dict, collection_name: str):
        qdrant_id = str(uuid.uuid5(self.namespace, str(payload["document_id"])))
        self._check_collection(collection_name)
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=qdrant_id,
                    vector=vector,
                    payload=payload,  # store the entire extraction + summary as payload
                )
            ],
        )

    def query(self, query: str, collection_name: str, top_k: int = 5) -> list[dict]:
        # This is a placeholder implementation; in a real implementation, you would first embed the query using the same embedding model
        # and then perform a vector search against the collection. For simplicity, we'll just return an empty list here.
        retval = self.client.query(query_text=query, collection_name=collection_name, limit=top_k)
        return retval.points


