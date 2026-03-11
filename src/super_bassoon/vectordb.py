from super_bassoon.op import get_secret
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, QueryResponse
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

    def upsert_batch(self, collection_name: str, points: list):
        self._check_collection(collection_name)
        self.client.upsert(
            collection_name=collection_name,
            points=[ PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points ]
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

    def query(self, query: str, collection_name: str, filters:Filter, top_k: int = 5) -> QueryResponse:
        results = self.client.query_points(
            collection_name=collection_name,
            query=query,  # In a real implementation, you would embed the query string into a vector using the same embedding model
            query_filter=filters,  # Apply filters to narrow down search results based on metadata
            limit=top_k
        )
        return results

    def query2(self, query: str, collection_name: str, top_k: int = 5) -> QueryResponse:
        results = self.client.query_points(
            collection_name=collection_name,
            query=query,  # In a real implementation, you would embed the query string into a vector using the same embedding model
            limit=top_k
        )
        return results


