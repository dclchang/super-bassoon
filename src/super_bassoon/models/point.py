import peewee
from .base import BaseModel, db, JsonField
from typing import cast

class Point(BaseModel):
    #id = peewee.CharField(primary_key=True)  # use a UUID string as the primary key
    document_id = peewee.IntegerField()  # reference to the source document
    point_id = peewee.CharField(max_length=100)  # e.g. "summary", "question_1", "question_2"
    point_id_uuid = peewee.CharField(max_length=36, unique=True)  # UUID generated from document_id + point_id for Qdrant
    point_type = peewee.CharField(max_length=50)  # e.g. "summary", "question"
    text = peewee.TextField()  # the original text that was embedded

    class Meta:
        table_name = "points"
