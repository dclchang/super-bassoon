import requests
from litellm import LiteLLM


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
