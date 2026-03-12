import asyncio
import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Optional

import rapidfuzz

from super_bassoon.models.base import db
from super_bassoon.models.document import Document
from super_bassoon.op import get_secret
from super_bassoon.paperless import PaperlessNgx


class Retriever:
    def __init__(self, paperless: PaperlessNgx):
        self.paperless = paperless

    def _hash_content(self, content: str) -> str:
        # Simple hash function for demonstration; replace with a proper hash in production
        return str(abs(hash(content)))

    def _generate_hash(self, text: str) -> str:
            """Creates a SHA-256 fingerprint of the text to detect changes."""
            if not text:
                return ""
            return hashlib.sha256(text.encode('utf-8')).hexdigest()

    async def retrieve(self):
            """
            source_documents: List of dicts [{'id': 4923, 'path': '...', 'ocr': '...'}]
            """
            new_count = 0
            update_count = 0
            skipped_count = 0

            document_type_ids = await self.paperless.get_document_types()
            document_types = [ dt['name'] for dt in document_type_ids ]

            for dt in document_types:
                document_ids = await self.paperless.get_document_ids_by_type(dt)
                source_documents = [await self.paperless.get_document(doc_id) for doc_id in document_ids]

                # db.atomic() wraps everything in one transaction. 
                # This is CRITICAL for performance when syncing 2,000+ docs.
                with db.atomic():
                    for doc in source_documents:
                        doc_id = doc['document_id']
                        txt = json.dumps(doc)  # Convert the entire document dict to a string for hashing
                        new_hash = self._generate_hash(txt)

                        # 1. Attempt to retrieve the existing record by Primary Key
                        # .get_or_none() is the safest way to check existence in Peewee
                        record = Document.get_or_none(Document.id == doc_id)

                        if record is None:
                            # Case: New Document
                            Document.create(
                                id=doc_id,
                                document_type=dt,
                                content=txt,
                                hash=new_hash,
                                status='pending'
                            )
                            new_count += 1
                        
                        elif record.hash != new_hash:
                            # Case: Existing document, but content has changed
                            record.content = txt
                            record.hash = new_hash
                            record.document_type = dt  # Update document type in case it changed
                            record.status = 'pending'  # Reset so Consumer re-processes it
                            record.updated_at = datetime.datetime.now()
                            record.save()
                            update_count += 1
                        
                        else:
                            # Case: Document exists and content is identical
                            skipped_count += 1

            print(f"--- Producer Sync Results ---")
            print(f"Queued for processing: {new_count}")
            print(f"Queued for update:     {update_count}")
            print(f"Skipped (no change):   {skipped_count}")
            print(f"-----------------------------")

    def _get_resolvable_fields(self, document_type: str) -> list[str]:
        schema_dir = Path(__file__).parent / "schemas"
        schema_file = schema_dir / f"{document_type}.txt"
        if not schema_file.exists():
            raise FileNotFoundError(f"Missing schema file: {document_type}")

        resolvable = []
        with open(schema_file) as f:
            for line in f:
                if '[resolvable]' in line:
                    match = re.match(r'\s*-\s*"(\w+)"', line)
                    if match:
                        resolvable.append(match.group(1))
        return resolvable
    
    def extract_filter_value(self, filter: dict, field: str) -> Optional[str]:
        """Extract the raw value for a field from the filter dict, if present."""
        for clause in ["must", "should", "must_not"]:
            if clause not in filter:
                continue
            for condition in filter[clause]:
                if condition.get("key") == field:
                    match = condition.get("match", {})
                    # handles both {"value": "Apple"} and {"any": ["Apple"]}
                    return match.get("value") or (match.get("any") or [None])[0]
        return None

    def get_distinct_values(self, field: str) -> list[str]:    
        cursor = db.execute_sql(
            f"SELECT DISTINCT json_extract(structured_content, '$.{field}') FROM documents WHERE structured_content IS NOT NULL"
        )
        return [row[0] for row in cursor.fetchall() if row[0]]

    def resolve_filter_field(self, filter: dict, field: str, resolved_values: list[str]) -> dict:
        """
        Replace a field's match value in the filter dict with rapidfuzz-resolved values.
        Works for any field: vendor, correspondent, recipient_name, etc.
        """
        if not resolved_values:
            for clause in ["must", "should", "must_not"]:
                if clause in filter:
                    filter[clause] = [
                        c for c in filter[clause]
                        if c.get("key") != field
                    ]
            return filter

        for clause in ["must", "should", "must_not"]:
            if clause not in filter:
                continue
            for condition in filter[clause]:
                if condition.get("key") == field:
                    if len(resolved_values) == 1:
                        condition["match"] = {"value": resolved_values[0]}
                    else:
                        condition["match"] = {"any": resolved_values}

        return filter

    def refine_filter(self, filter: dict, document_type: str) -> dict:
        # Placeholder for future implementation of LLM-based filter refinement
        resolvables = self._get_resolvable_fields(document_type)
        for resolvable in resolvables:
            key = self.extract_filter_value(filter, resolvable)
            if key is None:
                continue

            values = self.get_distinct_values(resolvable)

            if key in values:
                filter = self.resolve_filter_field(filter, resolvable, [key])
                continue

            matches = rapidfuzz.process.extract(key, values, scorer=rapidfuzz.fuzz.WRatio, limit=5)
            top_matches = [value for value, score, idx in matches if score > 80]
            filter = self.resolve_filter_field(filter, resolvable, top_matches)
        return filter
    
async def main():
    paperless = PaperlessNgx(
            base_url="http://192.168.68.222:8000", 
            api_key=get_secret("op://homelab/paperless-api-token/credential"))
    retriever = Retriever(paperless=paperless)
    await retriever.retrieve()
    await paperless.close()


if __name__ == "__main__":
    asyncio.run(main())

