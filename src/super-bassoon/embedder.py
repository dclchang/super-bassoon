import json
from models.document import Document
from models.base import db
from op import get_secret
from llmproxy import LlmProxy
from paperless import PaperlessNgx
from vectordb import VectorDb

class Embedder:
    def __init__(self, llmproxy: LlmProxy, vectordb: VectorDb, extractor_model: str, review_model: str, embedding_model: str):
        self.llm = llmproxy
        self.vectordb = vectordb
        self.extractor_model = extractor_model
        self.review_model = review_model
        self.embedding_model = embedding_model

    def consume(self):
        dt = "receipt"  # for now, hardcode to just process receipts; could be made dynamic later
        pending_docs = Document.select().where((Document.status == 'pending') & (Document.type == dt))
        for record in pending_docs:
            print(f"Processing document ID {record.id} of type {record.type}...")
            with db.atomic():
                record.status = "processing"
                record.save()

            content = record.content
            extraction = self.llm.extract(
                model=self.extractor_model,
                document=json.loads(content),  # Convert the string back to a dict for processing
                document_type=dt,
            )

            review = self.llm.review(model=self.review_model, extracted=extraction, document_type=record.type)
            summary = self.llm.summarise(model=self.extractor_model, extracted=extraction, document_type=record.type)

            vector = self.llm.vectorise(model=self.embedding_model, text=summary)
            self.vectordb.upsert(vector=vector, payload=extraction, collection_name=f"{record.type}_collection")

            with db.atomic():
                record.status = "processed"
                record.score = review["score"]  # store the review score in the DB for future reference
                record.score_reason = json.dumps(review["issues"])  # store the review issues as JSON string in the DB
                record.structured_content = extraction  # store the entire extracted JSON in the DB for future reference
                record.summary = summary  # store the summary in the DB for future reference
                record.save()





if __name__ == "__main__":
    paperless = PaperlessNgx(
        base_url="http://192.168.68.222:8000", 
        api_key=get_secret("op://homelab/paperless-api-token/credential"))
    
    llmproxy = LlmProxy(
        base_url="http://192.168.68.222:4040",
        api_key=get_secret("op://homelab/litellm-virtual-key-for-claude-code/credential"))

    vectordb = VectorDb(base_url="http://192.168.68.222:6333")
    consumer = Embedder(
        llmproxy=llmproxy,
        vectordb=vectordb,
        extractor_model="openai/claude-gemini-12",
        review_model="openai/falcon-7b",
        embedding_model="openai/nomic-embed-text"
    )
    consumer.consume()

