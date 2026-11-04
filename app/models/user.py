from typing import Literal

from sqlalchemy import ForeignKey, String, true
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

UserTypeName = Literal["recruiter", "candidate"]


class UserType(Base):
    __tablename__ = "user_types"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)


class User(Base):
    __tablename__ = "users"  # "user" is a reserved word in Postgres

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(default=True, server_default=true())
    user_type_id: Mapped[int] = mapped_column(ForeignKey("user_types.id"))

    user_type: Mapped[UserType] = relationship(lazy="joined")
