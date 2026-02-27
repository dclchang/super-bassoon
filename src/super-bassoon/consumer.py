import datetime
import hashlib
from models.document import Document
from models.base import db
from op import get_secret
from llmproxy import LlmProxy
from vectordb import VectorDb

class Consumer:
    def __init__(self, llm_proxy_url: str, llm_proxy_api_key: str, vector_db_url: str, extractor_model: str, review_model: str, embedding_model: str):
        self.llm = LlmProxy(llm_proxy_url, llm_proxy_api_key)
        self.vector_db_url = vector_db_url
        self.extractor_model = extractor_model
        self.review_model = review_model
        self.embedding_model = embedding_model

    def consume(self):
        pending_docs = Document.select().where((Document.status == 'pending') & (Document.document_type == 'receipt'))
        for record in pending_docs:
            with db.atomic():
                record.status = "processing"
                record.save()

            doc_id = record.document_id
            doc_type = record.document_type
            content = record.content

            print(f"Processing document ID {doc_id}...")

            # 1. Extract structured data using the LLM
            extraction = self.llm.extract(model=self.extractor_model, document=)

            extraction = self.llm.extract(model=self.extractor_model, text=content, document_type=doc_type)

            # 2. Review the extraction for quality; if it's below a threshold, skip vectorization
            score = self.llm.review(model=self.review_model, extracted=extraction, document_type=doc_type)

            summary = self.llm.summarise(model=self.extractor_model, extracted=extraction, document_type=doc_type)

            vector = self.llm.embed(model=self.embedding_model, text=summary)
            vectordb = VectorDb(url=self.vector_db_url, collection_name=f"{doc_type}_collection")
            vectordb.upsert(vector=vector, payload=extraction)





if __name__ == "__main__":
    consumer = Consumer(
        llm_proxy_url="http://192.168.68.222:4040",
        llm_proxy_api_key=get_secret("op://homelab/litellm-virtual-key-for-claude-code/credential"),
        extractor_model="openai/claude-gemini-12",
        review_model="openai/falcon-7b",
        embedding_model="openai/nomic-embed-text",
        vector_db_url="http://192.168.68.222:6333",
    )
    consumer.consume()

