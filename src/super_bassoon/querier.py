import asyncio
import logging
from super_bassoon.paperless import PaperlessNgx
from super_bassoon.llmproxy import LlmProxy
from super_bassoon.otel import Otel
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from super_bassoon.op import get_secret

class Querier:
    def __init__(self, llmproxy: LlmProxy, vectordb: QdrantClient, paperless: PaperlessNgx, logger: logging.Logger):
        self.llm = llmproxy
        self.vectordb = vectordb
        self.paperless = paperless
        self.logger = logger

    async def query(self, query: str, top_k: int=5) -> list[ScoredPoint]:
        self.logger.info(">>> Step 1: vectorise()...")
        vector = await self.llm.vectorise(text=query)
        self.logger.info(">>> Step 2: get_document_types()...")
        document_types = [ dt["name"] for dt in await self.paperless.get_document_types() ]
        self.logger.info(">>> Step 3: query_classifier()...")
        classifications = await self.llm.query_classifier(query=query, document_types=document_types)

        self.logger.info(">>> Step 4: get_top_k()...")
        k = await self.llm.get_top_k(query=query, document_type=classifications[0] if classifications else "receipt")
        self.logger.info(">>> Step 5: get_filters()...")
        filter = await self.llm.get_filters(query=query, document_types=classifications)
        self.logger.info(">>> Step 6: query_points()...")
        results = await asyncio.to_thread(
            self.vectordb.query_points,
            collection_name="my_collection",
            query=vector, query_filter=filter,
            limit=10, score_threshold=0.7
        )
        rets = [ f for f in results.points if f.score >= 0.7 ]
        self.logger.info(f">>> Done! Found {len(rets)} results")
        return rets


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

    llm = LlmProxy(base_url=get_secret("op://homelab/litellm-virtual-key-for-rag-app/url"),
                   api_key=get_secret("op://homelab/litellm-virtual-key-for-rag-app/credential"),
                   models={
                    "extractor": "openai/minimax-2.5",
                    "reviewer": "openai/falcon-7b",
                    "embedding": "openai/nomic-embed-text"
        })
    client = QdrantClient(url="http://192.168.68.222:6333")

    querier = Querier(llmproxy=llm, vectordb=client, paperless=paperless, logger=logger)
    question = "When did I buy the Sony Bravia?"
    results = await querier.query(question)
    answer = await querier.llm.answer_question(question, results)
    print(f"Answer: {answer}")
    client.close()

if __name__ == "__main__":
    asyncio.run(main())
