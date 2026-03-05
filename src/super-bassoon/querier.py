import asyncio
from paperless import PaperlessNgx
from llmproxy import LlmProxy
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from op import get_secret

class Querier:
    def __init__(self, llmproxy: LlmProxy, vectordb: QdrantClient, paperless: PaperlessNgx):
        self.llm = llmproxy
        self.vectordb = vectordb
        self.paperless = paperless

    async def query(self, query: str, top_k: int=5) -> list[ScoredPoint]:
        vector = await self.llm.vectorise(text=query)
        document_types = [ dt["name"] for dt in await self.paperless.get_document_types() ]
        classification = await self.llm.query_classifier(query=query, document_types=document_types)
        
        filters = await self.llm.query_filters(query=query, document_type=classification)
        results = await asyncio.to_thread(
            self.vectordb.query_points,
            collection_name=f"{classification}_collection",
            query=vector, query_filter=filters,
            limit=top_k
        )
        return results.points        


async def main():
    paperless = PaperlessNgx(
        base_url="http://192.168.68.222:8000",
        api_key=get_secret("op://homelab/paperless-api-token/credential"))

    llm = LlmProxy(base_url="http://192.168.68.222:4040",
                   api_key=get_secret("op://homelab/litellm-virtual-key-for-rag-app/credential"),
                   models={
                    "extractor": "openai/claude-gemini-12",
                    "reviewer": "openai/falcon-7b",
                    "embedding": "openai/nomic-embed-text"
        })
    client = QdrantClient(url="http://192.168.68.222:6333")

    querier = Querier(llmproxy=llm, vectordb=client, paperless=paperless)
    results = await querier.query("I remember going to a clinic at Anchorvale. How much did I pay for the consultation?")    
    client.close()

if __name__ == "__main__":
    asyncio.run(main())
