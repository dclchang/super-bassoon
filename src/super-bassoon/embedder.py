import json
from models.document import Document
from models.base import db
from op import get_secret
from llmproxy import LlmProxy
from paperless import PaperlessNgx
from vectordb import VectorDb

class Embedder:
    def __init__(self, llmproxy: LlmProxy, vectordb: VectorDb):
        self.llm = llmproxy
        self.vectordb = vectordb

    def embed(self):
        dt = "receipt"  # for now, hardcode to just process receipts; could be made dynamic later
        pending_documents = Document.select().where((Document.status == 'pending') & (Document.type == dt))
        for document in pending_documents:
            if document.id not in [222, 1896, 645, 681, 137]:
                continue
            print(f"Processing document ID {document.id} of type {document.type}...")
            with db.atomic():
                document.status = "processing"
                document.save()

            content = document.content
            extraction = self.llm.extract(
                document=json.loads(content),  # Convert the string back to a dict for processing
                document_type=dt,
            )

            review = self.llm.review(extracted=extraction, document_type=document.type)
            summary = self.llm.summarise(extracted=extraction, document_type=document.type)

            vector = self.llm.vectorise(text=summary)
            self.vectordb.upsert(vector=vector, payload=extraction, collection_name=f"{document.type}_collection")

            with db.atomic():
                document.status = "processed"
                document.score = review["score"]  # store the review score in the DB for future reference
                document.score_reason = json.dumps(review["issues"])  # store the review issues as JSON string in the DB
                document.structured_content = extraction  # store the entire extracted JSON in the DB for future reference
                document.summary = summary  # store the summary in the DB for future reference
                document.save()





if __name__ == "__main__":
    paperless = PaperlessNgx(
        base_url="http://192.168.68.222:8000", 
        api_key=get_secret("op://homelab/paperless-api-token/credential"))
    
    llmproxy = LlmProxy(
        base_url="http://192.168.68.222:4040",
        api_key=get_secret("op://homelab/litellm-virtual-key-for-claude-code/credential"),
        models={
            "extractor": "openai/claude-gemini-12",
            "reviewer": "openai/falcon-7b",
            "embedding": "openai/nomic-embed-text"
        })

    vectordb = VectorDb(base_url="http://192.168.68.222:6333")
    consumer = Embedder(
        llmproxy=llmproxy,
        vectordb=vectordb,
    )
    consumer.embed()

