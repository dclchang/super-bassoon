from op import get_secret
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

class VectorDb:
    def __init__(self, url: str, collection_name: str):
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.namespace = uuid.UUID(get_secret("op://homelab/qdrant-namespace/credential"))
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

    def upsert(self, vector: list[float], payload: dict):
        qdrant_id = str(uuid.uuid5(self.namespace, str(payload["document_id"])))
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=qdrant_id,
                    vector=vector,
                    payload=payload,  # store the entire extraction + summary as payload
                )
            ],
        )

