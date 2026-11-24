from fastapi import APIRouter

from app.api.v1.endpoints import agents, auth, documents, health, jds, qa, retriver

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(retriver.router, prefix="/retriever", tags=["retriever"])
api_router.include_router(qa.router, prefix="/qa", tags=["qa"])
api_router.include_router(jds.router, prefix="/jds", tags=["job descriptions"])
