import subprocess
import requests
from pathlib import Path


def get_secret(reference: str) -> str:
    result = subprocess.run(["op", "read", reference], capture_output=True, text=True, check=True)
    return result.stdout.strip()


class PaperlessNGX:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {token}"})

    def _get_all_pages(self, url: str, params: dict = None) -> list:
        results = []
        params = params or {}
        while url:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            results.extend(data.get("results", []))
            url = data.get("next")
            params = {}  # next URL already includes query params
        return results

    def get_document_types(self) -> list[dict]:
        """Return a list of all document types."""
        url = f"{self.base_url}/api/document_types/"
        return self._get_all_pages(url)

    def get_document_ids_by_type(self, document_type_id: int) -> list[int]:
        """Return all document IDs that belong to the given document type ID."""
        url = f"{self.base_url}/api/documents/"
        documents = self._get_all_pages(url, params={"document_type__id": document_type_id})
        return [doc["id"] for doc in documents]

    def get_document(self, document_id: int) -> dict:
        """Return the JSON content of a document given its ID."""
        url = f"{self.base_url}/api/documents/{document_id}/"
        response = self.session.get(url)
        response.raise_for_status()
        js = response.json()
        js["document_id"] = document_id  # add ID to the returned JSON for convenience
        return js

class LiteLLM:
    def __init__(self, base_url: str, model: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def extract(self, prompt: str, document_type: str) -> dict:
        prompts_dir = Path(__file__).parent / "prompts"
        prompt_file = next(prompts_dir.glob(f"{document_type}.txt"), None)
        if prompt_file is None:
            raise FileNotFoundError(f"No prompt file found for document type: {document_type}")
        system = prompt_file.read_text()
        return self.chat(prompt=prompt, system=system)

    def chat(self, prompt: str, system: str = None) -> str:
        """Send a prompt and return the assistant's reply as a string."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={"model": self.model, "messages": messages},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    PAPERLESS_URL = "http://192.168.68.222:8000"
    PAPERLESS_TOKEN = get_secret("op://homelab/paperless-api-token/credential")

    LITELLM_URL = "http://192.168.68.222:4040"
    LITELLM_MODEL = "claude-gemini-12"
    LITELLM_API_KEY = get_secret("op://homelab/litellm-virtual-key-for-claude-code/credential")

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
    litellm = LiteLLM(LITELLM_URL, LITELLM_MODEL, LITELLM_API_KEY)
    try:
        extraction_result = litellm.extract(prompt=receipt["content"], document_type="receipt")
        print(extraction_result)
    except Exception as exc:
        print("failed to extract with LiteLLM:", exc)
    #reply = litellm.chat("What is the capital of France?", system="You are a helpful assistant.")
    #print("LiteLLM reply:", reply)


