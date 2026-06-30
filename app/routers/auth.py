import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, EmailStr
from typing import Annotated
from fastapi import Depends, Request
from sqlalchemy.orm import Session
from passlib.context import CryptContext
# comes with python-multipart package
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, JSONResponse

try:
    from database import get_db
    from models import User
except ModuleNotFoundError:
    from app.database import get_db
    from app.models import User

load_dotenv()

router = APIRouter(
    prefix="/auth",
    tags=["auth"]
)

# run openssl rand -hex 32 in terminal to generate a secret key
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")


bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_bearer = OAuth2PasswordBearer(tokenUrl="auth/token")

APP_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=APP_DIR / "templates")

# Pages

from fastapi.responses import RedirectResponse

@router.get("/login")
async def login_page(request: Request):

    token = request.cookies.get("access_token")

    if token:
        # await get_current_user(token)
        return RedirectResponse(
            url="/chat",
            status_code=status.HTTP_302_FOUND
        )

    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"request": request},
    )

@router.get("/register")
async def register_page(request: Request):

    token = request.cookies.get("access_token")

    token = request.cookies.get("access_token")

    if token:
        return RedirectResponse(url="/chat", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse(
        request=request,
        name="register.html",
        context={"request": request},
    )

# EndPoints


# Depends - it is a dependency injection
db_dependency = Annotated[Session, Depends(get_db)]

class CreateUserRequest(BaseModel):
    email: EmailStr = Field(min_length=3)
    username: str = Field(min_length=3)
    password: str = Field(min_length=8)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

def authenticate_user(username: str, password: str, db: Session):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not bcrypt_context.verify(password, user.hashed_password):
        return False
    return user

def create_access_token(username: str, user_id: int, expires_delta: timedelta):
    to_encode = {"username": username, "id": user_id}
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]):
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("username")
        user_id: int = payload.get("id")
        if username is None or user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        return {"username": username, "id": user_id}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

@router.post('/', status_code=status.HTTP_201_CREATED)
def create_user(db: db_dependency, user: CreateUserRequest):
    # **user.dict() cannot be used as CreateUserRequest doesn't hvae hashed_password it has password
    existing = db.query(User).filter(
        User.email == user.email
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    create_user_model = User(
        email=user.email,
        username=user.username,
        hashed_password=bcrypt_context.hash(user.password)
    )
    db.add(create_user_model)
    db.commit()
    return {"message": "User created successfully"}


# @router.post('/token', response_model=TokenResponse)
# def get_access_token(db: db_dependency, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    
#     user = authenticate_user(form_data.username, form_data.password, db)
    
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Could not validate credentials"
#         )
    
#     token = create_access_token(user.username, user.id, timedelta(minutes=20))

#     return {"access_token": token, "token_type": "bearer"}


@router.post("/token")
def get_access_token(
    db: db_dependency,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
):
    user = authenticate_user(form_data.username, form_data.password, db)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    token = create_access_token(
        user.username,
        user.id,
        timedelta(minutes=20)
    )

    response = JSONResponse(
        content={"message": "Login successful"}
    )

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax"
    )

    return response