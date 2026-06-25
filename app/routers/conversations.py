from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, selectinload

try:
    from database import get_db
    from models import Conversation, Message
    from routers.auth import get_current_user
except ModuleNotFoundError:
    from app.database import get_db
    from app.models import Conversation, Message
    from app.routers.auth import get_current_user


router = APIRouter(
    prefix="/conversations",
    tags=["conversations"]
)

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]


class ConversationCreate(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=255)


class ConversationUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: str


class ConversationResponse(BaseModel):
    id: int
    thread_id: str
    title: str
    created_at: str


class ConversationDetailResponse(ConversationResponse):
    messages: list[MessageResponse]


def _conversation_or_404(db: Session, conversation_id: int, user_id: int) -> Conversation:
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user_id
    ).first()
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    return conversation


def _message_response(message: Message) -> MessageResponse:
    return MessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at.isoformat()
    )


def _conversation_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        id=conversation.id,
        thread_id=conversation.thread_id,
        title=conversation.title,
        created_at=conversation.created_at.isoformat()
    )


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    request: ConversationCreate,
    user: user_dependency,
    db: db_dependency
):
    conversation = Conversation(
        user_id=user["id"],
        title=request.title.strip()
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return _conversation_response(conversation)


@router.get("", response_model=list[ConversationResponse])
def list_conversations(user: user_dependency, db: db_dependency):
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user["id"]
    ).order_by(Conversation.created_at.desc()).all()
    return [_conversation_response(conversation) for conversation in conversations]


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(
    conversation_id: int,
    user: user_dependency,
    db: db_dependency
):
    conversation = db.query(Conversation).options(
        selectinload(Conversation.messages)
    ).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user["id"]
    ).first()
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    messages = sorted(conversation.messages, key=lambda item: item.created_at)
    return ConversationDetailResponse(
        **_conversation_response(conversation).dict(),
        messages=[_message_response(message) for message in messages]
    )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
def update_conversation(
    conversation_id: int,
    request: ConversationUpdate,
    user: user_dependency,
    db: db_dependency
):
    conversation = _conversation_or_404(db, conversation_id, user["id"])
    conversation.title = request.title.strip()
    db.commit()
    db.refresh(conversation)
    return _conversation_response(conversation)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int,
    user: user_dependency,
    db: db_dependency
):
    conversation = _conversation_or_404(db, conversation_id, user["id"])
    db.delete(conversation)
    db.commit()
