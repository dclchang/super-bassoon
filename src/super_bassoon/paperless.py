import aiohttp
from typing import Optional, List, Dict


class PaperlessNgx:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Token {self.api_key}"}
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_all_pages(self, url: str, params: dict = {}) -> List[Dict]:
        results = []
        params = params or {}
        session = await self._get_session()
        while url:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
            results.extend(data.get("results", []))
            url = data.get("next")
            params = {}
        return results

    async def get_document_types(self) -> List[Dict]:
        """Return a list of all document types."""
        url = f"{self.base_url}/api/document_types/"
        return await self._get_all_pages(url)

    async def get_document_ids_by_type(self, document_type: str) -> List[int]:
        """Return all document IDs that belong to the given document type ID."""
        document_types = await self.get_document_types()
        type_id = next((dt["id"] for dt in document_types if dt["name"] == document_type), None)
        if type_id is None:
            raise RuntimeError(f"No document type named '{document_type}' found in PaperlessNGX")
        url = f"{self.base_url}/api/documents/"
        documents = await self._get_all_pages(url, params={"document_type__id": type_id})
        return [doc["id"] for doc in documents]

    async def get_document(self, document_id: int) -> Dict:
        """Return the JSON content of a document given its ID."""
        session = await self._get_session()
        url = f"{self.base_url}/api/documents/{document_id}/"
        async with session.get(url) as response:
            response.raise_for_status()
            js = await response.json()
        js["document_id"] = document_id
        return js
