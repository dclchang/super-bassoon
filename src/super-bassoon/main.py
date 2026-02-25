from litellm import LiteLLM
from op import get_secret
from paperless import PaperlessNGX


# configuration constants used by example; could be made customizable later
PAPERLESS_URL = "http://192.168.68.222:8000"
PAPERLESS_TOKEN = get_secret("op://homelab/paperless-api-token/credential")

LITELLM_URL = "http://192.168.68.222:4040"
LITELLM_MODEL = "claude-gemini-12"
LITELLM_API_KEY = get_secret("op://homelab/litellm-virtual-key-for-claude-code/credential")


def main():
    ngx = PaperlessNGX(PAPERLESS_URL, PAPERLESS_TOKEN)
    document_types = ngx.get_document_types()
    receipt_id = next((dt["id"] for dt in document_types if dt["name"] == "receipt"), None)
    if receipt_id is None:
        raise RuntimeError("No document type named 'receipt' found in PaperlessNGX")

    receipts = ngx.get_document_ids_by_type(document_type_id=receipt_id)
    if not receipts:
        raise RuntimeError(f"No documents found for document type id {receipt_id}")

    # pick a sample receipt; make sure the index exists
    index = 7
    if index >= len(receipts):
        index = 0
    receipt = ngx.get_document(receipts[index])
    print(receipt)
    
    print("Extracting structured data from receipt using LiteLLM...")
    extractor = LiteLLM(LITELLM_URL, LITELLM_MODEL, LITELLM_API_KEY)
    try:
        extraction_result = extractor.extract(
            document=receipt,
            document_type="receipt",
        )
        print("Here we go:")
        print(extraction_result)

        reviewer = LiteLLM(LITELLM_URL, "falcon-7b", LITELLM_API_KEY)
        score = reviewer.review(extracted=extraction_result, document_type="receipt")
        print("")
        print(f"Review score: {score:.1f}/100")

    except Exception as exc:
        print("failed to extract with LiteLLM:", exc)


if __name__ == "__main__":
    main()