import datetime
import hashlib
import json
from models.document import Document
from models.base import db
from op import get_secret
from paperless import PaperlessNgx

class Retriever:
    #def __init__(self, paperless_url: str = None, paperless_token: str = None):
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

    def produce(self):
            """
            source_documents: List of dicts [{'id': 4923, 'path': '...', 'ocr': '...'}]
            """
            new_count = 0
            update_count = 0
            skipped_count = 0

            document_type_ids = self.paperless.get_document_types()
            document_types = [ dt['name'] for dt in document_type_ids ]

            for dt in document_types:
                document_ids = self.paperless.get_document_ids_by_type(dt)
                source_documents = [self.paperless.get_document(doc_id) for doc_id in document_ids]

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
                                type=dt,
                                content=txt,
                                hash=new_hash,
                                status='pending'
                            )
                            new_count += 1
                        
                        elif record.hash != new_hash:
                            # Case: Existing document, but content has changed
                            record.content = txt
                            record.hash = new_hash
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


if __name__ == "__main__":
    # Example usage
    paperless = PaperlessNgx(
         base_url="http://192.168.68.222:8000", 
         api_key=get_secret("op://homelab/paperless-api-token/credential"))
    producer = Retriever(paperless=paperless)
    producer.produce()

