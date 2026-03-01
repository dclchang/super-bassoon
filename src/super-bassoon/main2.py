from paperless import PaperlessNgx
from op import get_secret
from llmproxy import LlmProxy
from vectordb import VectorDb


if __name__ == "__main__":
    paperless = PaperlessNgx(
         base_url="http://192.168.68.222:8000",
         api_key=get_secret("op://homelab/paperless-api-token/credential")
    )
    llmproxy = LlmProxy(
        base_url="http://192.168.68.222:4040",
        api_key=get_secret("op://homelab/litellm-virtual-key-for-claude-code/credential"))
    
    vectordb = VectorDb(base_url="http://192.168.68.222:6333")

    query = "How much did I pay VicRoads for registration renewal?"
    document_types = [dt['name'] for dt in paperless.get_document_types()]
    document_type = llmproxy.query_classifier(model="openai/claude-gemini-12", query="When was my Aussie Broadband under $100?", document_types=document_types)
    filter = llmproxy.query_filters(model="openai/claude-gemini-12", query=query, document_type=document_type)
    vector = llmproxy.vectorise(model="openai/nomic-embed-text", text=query)
    #results = vectordb.query(query=vector, collection_name=f"{document_type}_collection", filters=filter, top_k=5)
    results = vectordb.query2(query=vector, collection_name=f"{document_type}_collection", top_k=5)
    for point in results.points:
        print(f"Document ID: {point.id}, Score: {point.score}, Payload: {point.payload}")



