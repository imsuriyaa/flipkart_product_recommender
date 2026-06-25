from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

try:
    from database import engine, Base
    from routers import auth, chat, conversations, users
except ModuleNotFoundError:
    from app.database import engine, Base
    from app.routers import auth, chat, conversations, users


BASE_DIR = Path(__file__).resolve().parent.parent

# Create all the tables in the database
Base.metadata.create_all(bind=engine)


app = FastAPI()

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get('/')
async def read_root(request: Request):
    return RedirectResponse(url="/auth/login-page", status_code=status.HTTP_302_FOUND)

@app.get("/healthy")
def health_check():
    return {"status": "Healthy"}

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(conversations.router)
app.include_router(chat.router)
