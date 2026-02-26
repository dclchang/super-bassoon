import datetime
import hashlib
import peewee
from models.document import Document
from models.base import db


class Producer:
    def __init__(self):
        pass

    def add_document(self, document_id: int, content: str):
        content_hash = self._hash_content(content)
        Document.create(document_id=document_id, content=content, content_hash=content_hash)

    def _hash_content(self, content: str) -> str:
        # Simple hash function for demonstration; replace with a proper hash in production
        return str(abs(hash(content)))

    def _generate_hash(self, text: str) -> str:
            """Creates a SHA-256 fingerprint of the text to detect changes."""
            if not text:
                return ""
            return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def sync_source(self, source_documents):
            """
            source_documents: List of dicts [{'id': 4923, 'path': '...', 'ocr': '...'}]
            """
            new_count = 0
            update_count = 0
            skipped_count = 0

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
                            ocr_text=doc['content'],
                            content_hash=new_hash,
                            status='pending'
                        )
                        new_count += 1
                    
                    elif record.content_hash != new_hash:
                        # Case: Existing document, but OCR has changed
                        record.ocr_text = doc['ocr']
                        record.content_hash = new_hash
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
    producer = Producer()
    sample_docs = [
        {'document_id': 1, 'content': 'This is the OCR text of document 1.'},
        {'document_id': 2, 'content': 'This is the OCR text of document 2.'},
    ]
    producer.sync_source(sample_docs)

