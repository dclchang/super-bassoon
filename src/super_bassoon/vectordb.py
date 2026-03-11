from super_bassoon.op import get_secret
from qdrant_client import QdrantClient
#from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, QueryResponse
import qdrant_client.models as qm
import uuid

class VectorDb:
    def __init__(self, base_url: str):
        self.client = QdrantClient(url=base_url)
        self.namespace = uuid.UUID(get_secret("op://homelab/qdrant-namespace/credential"))

    def _check_collection(self, collection_name: str):
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qm.VectorParams(size=768, distance=qm.Distance.COSINE),
            )

    def upsert_batch(self, collection_name: str, points: list):
        self._check_collection(collection_name)
        self.client.upsert(
            collection_name=collection_name,
            points=[ qm.PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"]) for p in points ]
        )

    def upsert(self, vector: list[float], payload: dict, collection_name: str):
        qdrant_id = str(uuid.uuid5(self.namespace, str(payload["document_id"])))
        self._check_collection(collection_name)
        self.client.upsert(
            collection_name=collection_name,
            points=[
                qm.PointStruct(
                    id=qdrant_id,
                    vector=vector,
                    payload=payload,  # store the entire extraction + summary as payload
                )
            ],
        )

