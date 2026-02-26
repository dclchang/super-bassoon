import datetime
import hashlib
from models.document import Document
from models.base import db
from op import get_secret
from paperless import PaperlessNGX

class Producer:
    def __init__(self, paperless_url: str = None, paperless_token: str = None):
        self.paperless_url = paperless_url
        self.paperless_token = get_secret(paperless_token)
        self.paperless = PaperlessNGX(self.paperless_url, self.paperless_token)

    # def add_document(self, document_id: int, content: str):
    #     content_hash = self._hash_content(content)
    #     Document.create(document_id=document_id, content=content, content_hash=content_hash)

    def _hash_content(self, content: str) -> str:
        # Simple hash function for demonstration; replace with a proper hash in production
        return str(abs(hash(content)))

    def _generate_hash(self, text: str) -> str:
            """Creates a SHA-256 fingerprint of the text to detect changes."""
            if not text:
                return ""
            return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def sync_source(self):
            """
            source_documents: List of dicts [{'id': 4923, 'path': '...', 'ocr': '...'}]
            """
            new_count = 0
            update_count = 0
            skipped_count = 0

            document_ids = self.paperless.get_document_ids_by_type("receipt")
            source_documents = [self.paperless.get_document(doc_id) for doc_id in document_ids]

            # db.atomic() wraps everything in one transaction. 
            # This is CRITICAL for performance when syncing 2,000+ docs.
            with db.atomic():
                for doc in source_documents:
                    doc_id = doc['document_id']
                    new_hash = self._generate_hash(doc['content'])

                    # 1. Attempt to retrieve the existing record by Primary Key
                    # .get_or_none() is the safest way to check existence in Peewee
                    record = Document.get_or_none(Document.document_id == doc_id)

                    if record is None:
                        # Case: New Document
                        Document.create(
                            document_id=doc_id,
                            #file_path=doc['path'],
                            content=doc['content'],
                            hash=new_hash,
                            status='pending'
                        )
                        new_count += 1
                    
                    elif record.hash != new_hash:
                        # Case: Existing document, but content has changed
                        record.content = doc['content']
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
    producer = Producer(
         paperless_url="http://192.168.68.222:8000", 
         paperless_token="op://homelab/paperless-api-token/credential")
    producer.sync_source()

