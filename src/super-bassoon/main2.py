import asyncio
from paperless import PaperlessNgx
from op import get_secret
from llmproxy import LlmProxy
from vectordb import VectorDb
from models.base import db
from retriever import Retriever
import rapidfuzz

async def main():
    paperless = PaperlessNgx(
         base_url="http://192.168.68.222:8000",
         api_key=get_secret("op://homelab/paperless-api-token/credential")
    )
    llmproxy = LlmProxy(
        base_url="http://192.168.68.222:4040",
        api_key=get_secret("op://homelab/litellm-virtual-key-for-rag-app/credential"),
        models={
            "extractor": "openai/claude-gemini-12",
            "reviewer": "openai/falcon-7b",
            "embedding": "openai/nomic-embed-text"
        })
    
    vectordb = VectorDb(base_url="http://192.168.68.222:6333")

    query = "What did I buy with a receipt number of 'INV-34183630'?"
    document_types = [dt['name'] for dt in await paperless.get_document_types()]
    document_type = await llmproxy.query_classifier(query=query, document_types=document_types)
    filter = await llmproxy.query_filters(query=query, document_type=document_type)

    retriever = Retriever(paperless=paperless)
    new_filter = retriever.refine_filter(filter=filter, document_type=document_type)

    cursor = db.execute_sql(
        "SELECT DISTINCT json_extract(structured_content, '$.vendor') FROM documents WHERE structured_content IS NOT NULL"
    )
    vendors = [row[0] for row in cursor.fetchall() if row[0]]
    matches = rapidfuzz.process.extract("Apple", vendors, scorer=rapidfuzz.fuzz.WRatio, limit=5)
    tm = [vendor for vendor, score, idx in matches if score > 80]




    vector = await llmproxy.vectorise(text=query)
    #results = vectordb.query(query=vector, collection_name=f"{document_type}_collection", filters=filter, top_k=5)
    results = vectordb.query2(query=vector, collection_name=f"{document_type}_collection", top_k=5)
    for point in results.points:
        print(f"Document ID: {point.id}, Score: {point.score}, Payload: {point.payload}")

    await paperless.close()

if __name__ == "__main__":
    asyncio.run(main())
