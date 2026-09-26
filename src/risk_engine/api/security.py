from fastapi.security import APIKeyHeader
from fastapi import Depends, HTTPException, Request

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(request: Request, key: str | None = Depends(api_key_header)) -> None:
    if key != request.app.state.settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
