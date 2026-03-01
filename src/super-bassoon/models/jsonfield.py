import json
import peewee

class JsonField(peewee.TextField):
    """Custom Peewee field to store JSON data as text."""
    
    def db_value(self, value):
        """Convert Python dict to JSON string before saving to DB."""
        if value is None:
            return None
        return json.dumps(value)

    def python_value(self, value):
        """Convert JSON string back to Python dict when loading from DB."""
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value  # If it's not valid JSON, return the raw value

