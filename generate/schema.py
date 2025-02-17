import json

from utils.model import Base

schema = Base.all_schema()

with open("generate/schema.json", "w", encoding="utf-8") as f:
    json.dump(schema, f, indent=4, ensure_ascii=False)
