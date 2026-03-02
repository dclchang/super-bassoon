import subprocess
import sys
from pathlib import Path

# ensure workspace root is on sys.path so we can import paperless
sys.path.insert(0, str(Path(__file__).parents[1]))

from paperless import PaperlessNGX, get_secret
from litellm import LiteLLM

# configuration (same as main script)
PAPERLESS_URL = "http://192.168.68.222:8000"
PAPERLESS_TOKEN = get_secret("op://homelab/paperless-api-token/credential")

LITELLM_URL = "http://192.168.68.222:4040"
LITELLM_MODEL = "claude-gemini-12"
LITELLM_API_KEY = get_secret("op://homelab/litellm-virtual-key-for-rag-app/credential")


def main():
    ngx = PaperlessNGX(PAPERLESS_URL, PAPERLESS_TOKEN)
    extractor = LiteLLM(LITELLM_URL, LITELLM_MODEL, LITELLM_API_KEY)
    reviewer = LiteLLM(LITELLM_URL, "falcon-7b", LITELLM_API_KEY)

    # gather first 15 receipts
    document_types = ngx.get_document_types()
    receipt_id = next((dt["id"] for dt in document_types if dt.get("name") == "receipt"), None)
    if receipt_id is None:
        raise RuntimeError("No document type named 'receipt'")

    ids = ngx.get_document_ids_by_type(document_type_id=receipt_id)
    ids = ids[:2]

    results = []  # tuples of (document_id, score)

    for doc_id in ids:
        doc = ngx.get_document(doc_id)
        data = extractor.extract(document=doc, document_type="receipt")
        score = reviewer.review(data, document_type="receipt")
        results.append((doc_id, score))

    # write markdown table
    out_path = Path(__file__).parent / "results.md"
    with open(out_path, "w") as f:
        f.write("| document_id | reviewer_score |\n")
        f.write("|-------------|----------------|\n")
        for doc_id, score in results:
            f.write(f"| {doc_id} | {score} |\n")

    print(f"results written to {out_path}")


if __name__ == "__main__":
    main()
