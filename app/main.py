from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.indexing_service import check_vector_db

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await check_vector_db()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.include_router(api_router, prefix="/api/v1")
