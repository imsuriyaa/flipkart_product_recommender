from os import getenv

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

SQLALCHEMY_DATABASE_URL = getenv(
    "DATABASE_URL",
    "sqlite:///./chatbot.db"
)
# SQLALCHEMY_DATABASE_URL = 'mysql+pymysql://root:@localhost:3306/TodoApplicationDatabase'

# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:Senthil3%40@localhost:5432/TodoApplicationDatabase"

# create a database engine - this is what we use to connect to the database
# multiple threads can access the database at the same time - check_same_thread=False

# sqlite only
# engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={'check_same_thread': False})

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)


# create a database session factory - this is what we use to create sessions to query the database
# autocommit=False - we need to commit our changes manually
# autoflush=False - we need to flush our changes manually
# bind=engine - this is what we use to connect to the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# create a database object that we can use to create our tables
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
