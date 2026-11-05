from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class VectorMigration(Base):
    """One row per Qdrant migration file that has been applied (like alembic_version)."""
    __tablename__ = "vector_migrations"

    id: Mapped[str] = mapped_column(String(200), primary_key=True)  # file name, e.g. 0001_create_resume_chunks_v1
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
