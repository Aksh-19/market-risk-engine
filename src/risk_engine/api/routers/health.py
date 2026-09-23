from fastapi import APIRouter, Request, Response

router = APIRouter(tags=["ops"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready(request: Request, response: Response):
    ok = getattr(request.app.state, "returns", None) is not None
    if not ok:
        response.status_code = 503
    return {"ready": ok}
