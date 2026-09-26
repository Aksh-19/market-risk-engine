import json
from pathlib import Path


def test_openapi_schema_matches_snapshot(client):
    current = client.app.openapi()
    baseline = json.loads(Path("tests/api/snapshots/openapi.json").read_text())
    assert current == baseline, (
        "OpenAPI schema changed. If intentional, rerun "
        "scripts/generate_openapi_snapshot.py and commit the new baseline."
    )
