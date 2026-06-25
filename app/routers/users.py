from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Annotated
from fastapi import Depends, HTTPException, status

try:
    from database import get_db
    from models import User
    from routers.auth import get_current_user
except ModuleNotFoundError:
    from app.database import get_db
    from app.models import User
    from app.routers.auth import get_current_user

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

# Depends - it is a dependency injection
db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]


# Endpoints


@router.get('/me')
def read_user_data(user: user_dependency, db: db_dependency):
    
    user_model = db.query(User).filter(User.id == user.get('id')).first()
    
    if user_model is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        'user_id': user_model.id,
        'username': user_model.username,
        'email': user_model.email
    }
