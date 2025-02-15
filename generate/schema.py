import json

from utils.model import Base

schema = Base.all_schema()

with open("generate/schema.json", "w") as f:
    json.dump(schema, f, indent=4)
