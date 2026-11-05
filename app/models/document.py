from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, UniqueConstraint, func, true
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

JsonType = JSON().with_variant(JSONB, "postgresql")  # JSONB on Postgres, JSON on SQLite (tests)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("generated_by", "document_hash"),)  # same file once per user

    id: Mapped[int] = mapped_column(primary_key=True)
    document_hash: Mapped[str] = mapped_column(String(64), index=True)  # SHA-256 of the file
    generated_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())  # upload time

    meta_data: Mapped["DocumentMetaData"] = relationship(back_populates="document")


class DocumentMetaData(Base):
    __tablename__ = "document_meta_data"

    id: Mapped[int] = mapped_column(primary_key=True)
    doc_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), unique=True)
    document_plain_data: Mapped[list] = mapped_column(JsonType)  # lines; list index = line number
    parsed_doc_json: Mapped[dict] = mapped_column(JsonType)

    document: Mapped[Document] = relationship(back_populates="meta_data")
