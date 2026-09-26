import json
from risk_engine.api.main import create_app

with open("tests/api/snapshots/openapi.json", "w") as f:
    json.dump(create_app().openapi(), f, indent=2, sort_keys=True)
