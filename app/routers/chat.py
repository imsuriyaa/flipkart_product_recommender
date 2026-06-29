from functools import lru_cache
from pathlib import Path
from typing import Annotated
from fastapi import APIRouter, Depends, Form, HTTPException, Request, status
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Conversation, Message
from app.routers.auth import get_current_user
from workflow.llm_chat import AgenticRAG
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage
)


router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: int | None = None
    thread_id: str | None = None

class ChatResponse(BaseModel):
    conversation_id: int
    thread_id: str
    user_message: str
    assistant_message: str

class ConversationSummary(BaseModel):
    id: int
    thread_id: str
    title: str
    created_at: str


@lru_cache(maxsize=1)
def get_chat_engine() -> AgenticRAG:
    return AgenticRAG()


def _make_title(message: str) -> str:
    title = " ".join(message.strip().split())
    if len(title) > 60:
        return f"{title[:57]}..."
    return title or "New conversation"


def _get_or_create_conversation(
    db: Session,
    user_id: int,
    request: ChatRequest,
) -> Conversation:

    conversation = None

    if request.conversation_id is not None:
        conversation = db.query(Conversation).filter(
            Conversation.id == request.conversation_id,
            Conversation.user_id == user_id
        ).first()
        if conversation is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
    elif request.thread_id:
        conversation = db.query(Conversation).filter(
            Conversation.thread_id == request.thread_id,
            Conversation.user_id == user_id
        ).first()

    if conversation:
        return conversation

    conversation = Conversation(
        user_id=user_id,
        title=_make_title(request.message)
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def _load_history(
    db: Session,
    conversation_id: int,
) -> list[BaseMessage]:
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    history = [
        HumanMessage(content=msg.content)
        if msg.role == "user"
        else AIMessage(content=msg.content)
        for msg in messages
    ]
    return history


@router.post("", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user: user_dependency,
    db: db_dependency,
    chat_engine: Annotated[AgenticRAG, Depends(get_chat_engine)]
):
    
    conversation = _get_or_create_conversation(db, user["id"], request)

    history = _load_history(db, conversation.id)

    user_message = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.message
    )
    db.add(user_message)
    
    
    try:
        assistant_text = chat_engine.run(
            request.message,
            history=history
        )
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM chat failed: {exc}"
        ) from exc

    assistant_message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=assistant_text
    )
    db.add(assistant_message)
    db.commit()

    return ChatResponse(
        conversation_id=conversation.id,
        thread_id=conversation.thread_id,
        user_message=request.message,
        assistant_message=assistant_text
    )



@router.get("/conversations", response_model=list[ConversationSummary])
def get_user_conversations(
    user: user_dependency,
    db: db_dependency,
):
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == user["id"])
        .order_by(Conversation.created_at.desc())
        .all()
    )

    return [
        ConversationSummary(
            id=conversation.id,
            thread_id=conversation.thread_id,
            title=conversation.title,
            created_at=conversation.created_at.isoformat(),
        )
        for conversation in conversations
    ]






