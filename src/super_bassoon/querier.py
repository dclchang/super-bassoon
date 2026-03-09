import asyncio
from super_bassoon.paperless import PaperlessNgx
from super_bassoon.llmproxy import LlmProxy
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from super_bassoon.op import get_secret

class Querier:
    def __init__(self, llmproxy: LlmProxy, vectordb: QdrantClient, paperless: PaperlessNgx):
        self.llm = llmproxy
        self.vectordb = vectordb
        self.paperless = paperless

    async def query(self, query: str, top_k: int=5) -> list[ScoredPoint]:
        print(">>> Step 1: vectorise()...")
        vector = await self.llm.vectorise(text=query)
        print(">>> Step 2: get_document_types()...")
        document_types = [ dt["name"] for dt in await self.paperless.get_document_types() ]
        print(">>> Step 3: query_classifier()...")
        classifications = await self.llm.query_classifier(query=query, document_types=document_types)
        
        print(">>> Step 4: get_top_k()...")
        k = await self.llm.get_top_k(query=query, document_type=classifications[0] if classifications else "receipt")
        print(">>> Step 5: get_filters()...")
        filter = await self.llm.get_filters(query=query, document_types=classifications)
        print(">>> Step 6: query_points()...")
        results = await asyncio.to_thread(
            self.vectordb.query_points,
            collection_name="my_collection",
            query=vector, query_filter=filter,
            limit=k
        )
        rets = [ f for f in results.points if f.score >= 0.7 ]
        print(f">>> Done! Found {len(rets)} results")
        return rets


async def main():
    paperless = PaperlessNgx(
        base_url="http://192.168.68.222:8000",
        api_key=get_secret("op://homelab/paperless-api-token/credential"))

    llm = LlmProxy(base_url="http://192.168.68.222:4040",
                   api_key=get_secret("op://homelab/litellm-virtual-key-for-rag-app/credential"),
                   models={
                    #"extractor": "gemini/gemini/gemini-2.5-flash",
                    #"extractor": "openai/claude-gemini-12",
                    "extractor": "openai/nous-hermes-2-pro",
                    "reviewer": "openai/falcon-7b",
                    "embedding": "openai/nomic-embed-text"
        })
    client = QdrantClient(url="http://192.168.68.222:6333")

    querier = Querier(llmproxy=llm, vectordb=client, paperless=paperless)
    #results = await querier.query("I remember going to a clinic at Anchorvale. How much did I pay for the consultation?")    
    results = await querier.query("How much have I paid for Aussie Broadband last year?")
    client.close()

if __name__ == "__main__":
    asyncio.run(main())
