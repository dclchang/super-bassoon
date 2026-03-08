import peewee
from .base import BaseModel, db, JsonField
from typing import cast

class Document(BaseModel):
    id = peewee.IntegerField(primary_key=True)
    document_type = peewee.CharField(max_length=100)  # e.g. "receipt", "invoice"
    document_sub_type = peewee.CharField(max_length=100, null=True)  # e.g. "grocery_receipt", "utility_invoice"
    created_at = peewee.DateTimeField(null=True)  # timestamp from PaperlessNGX
    added_at = peewee.DateTimeField(null=True)  # when we added this to our DB
    updated_at = peewee.DateTimeField(null=True)  # when we last updated this
    content = peewee.TextField()
    structured_content: dict = cast(dict, JsonField(null=True))  # optional field to store the extracted JSON from the document for future reference
    summary: str = cast(str, peewee.TextField(null=True))  # optional field to store LLM-generated summary
    hash = peewee.CharField(max_length=64)  # store a hash of the content for quick comparisons
    status: str = cast(str, peewee.CharField(max_length=20, default="new"))  # e.g. "new", "processed", "error"
    score: float = cast(float, peewee.FloatField(default=0))  # optional field to store review score from LLM
    score_reason: str = cast(str, peewee.TextField(null=True))  # optional field to store LLM's explanation of the score
    error: str = cast(str, peewee.TextField(null=True))  # optional field to store error messages if processing fails

    class Meta:
        table_name = "documents"

db.connect()
db.create_tables([Document], safe=True)
