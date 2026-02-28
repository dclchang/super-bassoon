import peewee
from .base import BaseModel, db

class Document(BaseModel):
    id = peewee.IntegerField(primary_key=True)
    type = peewee.CharField(max_length=100)  # e.g. "receipt", "invoice"
    created_at = peewee.DateTimeField(null=True)  # timestamp from PaperlessNGX
    added_at = peewee.DateTimeField(null=True)  # when we added this to our DB
    updated_at = peewee.DateTimeField(null=True)  # when we last updated this
    content = peewee.TextField()
    summary = peewee.TextField(null=True)  # optional field to store LLM-generated summary
    hash = peewee.CharField(max_length=64)  # store a hash of the content for quick comparisons
    status = peewee.CharField(max_length=20, default="new")  # e.g. "new", "processed", "error"
    score = peewee.FloatField(default=0)  # optional field to store review score from LLM
    score_reason = peewee.TextField(null=True)  # optional field to store LLM's explanation of the score

    class Meta:
        table_name = "documents"

db.connect()
db.create_tables([Document], safe=True)
