from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.security import create_access_token, hash_password, verify_password
from app.db.database import get_db
from app.models import User, UserType
from app.schemas.auth import LoginRequest, SignupRequest, TokenResponse, UserOut

router = APIRouter()


def to_out(user: User) -> UserOut:
    return UserOut(id=user.id, email=user.email, user_type=user.user_type.name, is_active=user.is_active)


@router.post("/signup", response_model=UserOut, status_code=201)
async def signup(body: SignupRequest, db: AsyncSession = Depends(get_db)):
    email = body.email.lower()
    if await db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "Email already registered")

    user_type = await db.scalar(select(UserType).where(UserType.name == body.user_type))
    user = User(email=email, hashed_password=hash_password(body.password), user_type=user_type)
    db.add(user)
    await db.commit()
    return to_out(user)


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(select(User).where(User.email == body.email.lower()))
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Invalid email or password")
    if not user.is_active:
        raise HTTPException(403, "Account is inactive")
    return TokenResponse(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return to_out(user)
