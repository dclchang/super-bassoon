import peewee

db = peewee.SqliteDatabase('super_bassoon.db')

class BaseModel(peewee.Model):
    class Meta:
        database = db