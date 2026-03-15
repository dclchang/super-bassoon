import asyncio
import json
import logging
import hashlib
import uuid
from super_bassoon.models.base import db
from super_bassoon.models.document import Document
from super_bassoon.models.point import Point
from super_bassoon.op import get_secret
from super_bassoon.llmproxy import LlmProxy
from super_bassoon.paperless import PaperlessNgx
from super_bassoon.vectordb import VectorDb
from super_bassoon.otel import Otel


class Embedder:
    def __init__(self, llmproxy: LlmProxy, vectordb: VectorDb, logger: logging.Logger):
        self.llm = llmproxy
        self.vectordb = vectordb
        self.logger = logger

    def _generate_id(self, source_id: int, point_type: str, index: int = 0) -> str:
        raw = f"{source_id}_{point_type}_{index}"
        return str(uuid.UUID(hashlib.md5(raw.encode()).hexdigest()))

    async def _process_document(self, document: Document) -> None:
        doc_type = str(document.document_type)

        with db.atomic():
            document.status = "processing"
            document.save()

        try:
            content = str(document.content)
            extraction = await self.llm.extract(document=json.loads(content), document_type=doc_type)

            #review = await self.llm.review(extracted=extraction, document_type=doc_type)
            summary = await self.llm.summarise(extracted=extraction, document_type=doc_type)
            questions = await self.llm.generate_questions(summary=summary)  # just added

            points = []
            summary_vector = await self.llm.vectorise(text=summary)
            points.append({
                "id": self._generate_id(source_id=document.id, point_type="summary"),   # type: ignore
                "vector": summary_vector,
                "payload": {
                    "document_id": document.id,
                    "document_type": doc_type,
                    "point_type": "summary",
                    "text": summary,
                }
            })

            Point.create(
                document_id=document.id,
                point_id=f"{document.id}_summary_{0}",
                point_id_uuid=self._generate_id(source_id=document.id, point_type="summary"),  # type: ignore
                point_type="summary",
                text=summary
            )

            for idx, question in enumerate(questions):
                question_vector = await self.llm.vectorise(text=question)
                points.append({
                    "id": self._generate_id(source_id=document.id, point_type="question", index=idx),   # type: ignore
                    "vector": question_vector,
                    "payload": {
                        "document_id": document.id,
                        "document_type": doc_type,
                        "point_type": "question",
                        "text": question,
                    }
                })
                Point.create(
                    document_id=document.id,
                    point_id=f"{document.id}_question_{idx}",
                    point_id_uuid=self._generate_id(source_id=document.id, point_type="question", index=idx),  # type: ignore
                    point_type="question",
                    text=question)


            self.vectordb.upsert_batch(collection_name="my_collection", points=points)

            with db.atomic():
                document.status = "processed"
                #document.score = review["score"]
                #document.score_reason = json.dumps(review["issues"])
                document.structured_content = extraction
                document.summary = summary
                document.save()

            self.logger.info(f"Completed document ID {document.id}")
        except Exception as e:
            self.logger.error(f"Error processing document {document.id}: {e}")
            with db.atomic():
                document.status = "failed"
                document.error = str(e)
                document.save()

    async def embed(self):
        dt = "receipt"
        pending_documents = list(
            Document.select().where(
                (Document.status.in_(['pending', ])) & (Document.document_type == dt))   # type: ignore
        )

        if not pending_documents:
            self.logger.info("No pending documents to process")
            return

        self.logger.info(f"Found {len(pending_documents)} pending documents, processing with semaphore throttling...")

        for doc in pending_documents:
            await self._process_document(doc)

        self.logger.info("All documents processed")

async def main():
    logger = Otel(
        service_name="super-bassoon",
        host=get_secret("op://homelab/grafana-otel-endpoint/url"),
        instance_id=get_secret("op://homelab/grafana-otel-endpoint/instance_id"),
        api_key=get_secret("op://homelab/grafana-otel-endpoint/credential"),
    )

    paperless = PaperlessNgx(
        base_url=get_secret("op://homelab/paperless-api-token/url"),
        api_key=get_secret("op://homelab/paperless-api-token/credential"))

    llmproxy = LlmProxy(
        base_url=get_secret("op://homelab/litellm-virtual-key-for-rag-app/url"),
        api_key=get_secret("op://homelab/litellm-virtual-key-for-rag-app/credential"),
        models={
            "extractor": "openai/qwen25-7",
            "reviewer": "openai/falcon-7b",
            "embedding": "openai/nomic-embed-text"
        },
        max_concurrent=2
    )

    vectordb = VectorDb(base_url="http://192.168.68.222:6333")
    embedder = Embedder(llmproxy=llmproxy, vectordb=vectordb, logger=logger)
    await embedder.embed()
    await paperless.close()

if __name__ == "__main__":
    asyncio.run(main())
