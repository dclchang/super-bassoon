import peewee
from .base import BaseModel, db

class Document(BaseModel):
    document_id = peewee.IntegerField(primary_key=True)
    #created = peewee.DateTimeField()
    #added = peewee.DateTimeField()
    content = peewee.TextField()
    hash = peewee.CharField(max_length=64)  # store a hash of the content for quick comparisons
    status = peewee.CharField(max_length=20, default="new")  # e.g. "new", "processed", "error"

    class Meta:
        table_name = "documents"

db.connect()
db.create_tables([Document], safe=True)
