import asyncio
import json

from super_bassoon.models.base import db
from super_bassoon.models.document import Document
from super_bassoon.op import get_secret
from super_bassoon.llmproxy import LlmProxy
from super_bassoon.paperless import PaperlessNgx
from super_bassoon.vectordb import VectorDb


class Embedder:
    def __init__(self, llmproxy: LlmProxy, vectordb: VectorDb):
        self.llm = llmproxy
        self.vectordb = vectordb

    async def _process_document(self, document) -> None:
        doc_id = document.id
        doc_type = document.document_type

        with db.atomic():
            document.status = "processing"
            document.save()

        try:
            content = document.content
            extraction = await self.llm.extract(document=json.loads(content), document_type=doc_type)

            review = await self.llm.review(extracted=extraction, document_type=doc_type)
            summary = await self.llm.summarise(extracted=extraction, document_type=doc_type)

            vector = await self.llm.vectorise(text=summary)
            self.vectordb.upsert(vector=vector, payload=extraction,
                collection_name="my_collection"
            )

            with db.atomic():
                document.status = "processed"
                document.score = review["score"]
                document.score_reason = json.dumps(review["issues"])
                document.structured_content = extraction
                document.summary = summary
                document.save()

            print(f"Completed document ID {doc_id}")
        except Exception as e:
            print(f"Error processing document {doc_id}: {e}")
            with db.atomic():
                document.status = "failed"
                document.error = str(e)
                document.save()

    async def embed(self):
        dt = "receipt"
        pending_documents = list(
            Document.select().where((Document.status in ['pending', 'processing']) & (Document.document_type == dt))
        )

        if not pending_documents:
            print("No pending documents to process")
            return

        print(f"Found {len(pending_documents)} pending documents, processing with semaphore throttling...")

        await asyncio.gather(*[self._process_document(doc) for doc in pending_documents])

        print("All documents processed")

async def main():
    paperless = PaperlessNgx(
        base_url="http://192.168.68.222:8000",
        api_key=get_secret("op://homelab/paperless-api-token/credential"))

    llmproxy = LlmProxy(
        base_url="http://192.168.68.222:4040",
        api_key=get_secret("op://homelab/litellm-virtual-key-for-rag-app/credential"),
        models={
            "extractor": "openai/claude-gemini-12",
            "reviewer": "openai/falcon-7b",
            "embedding": "openai/nomic-embed-text"
        },
        max_concurrent=2
    )

    vectordb = VectorDb(base_url="http://192.168.68.222:6333")
    consumer = Embedder(
        llmproxy=llmproxy,
        vectordb=vectordb,
    )
    await consumer.embed()
    await paperless.close()

if __name__ == "__main__":
    asyncio.run(main())
