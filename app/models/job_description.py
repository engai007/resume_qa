from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.models.document import JsonType


class JobDescription(Base):
    """An uploaded job description: the text the LLM read + the structure it extracted. See docs/jd_parser_plan.md."""
    __tablename__ = "job_descriptions"
    __table_args__ = (UniqueConstraint("created_by", "text_hash"),)  # same JD text once per user

    id: Mapped[int] = mapped_column(primary_key=True)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    text_hash: Mapped[str] = mapped_column(String(64))  # SHA-256 of the normalised text (not the file)
    filename: Mapped[str] = mapped_column(String(255))
    raw_text: Mapped[str] = mapped_column(Text)  # what the LLM read: re-parse later without the file
    parsed_json: Mapped[dict] = mapped_column(JsonType)  # the JDStructure
    model_id: Mapped[str] = mapped_column(String(200))  # which model parsed it
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
