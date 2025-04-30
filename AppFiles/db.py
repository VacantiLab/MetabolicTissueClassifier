from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use SQLite for simplicity; swap with your MySQL URL if needed
SQLALCHEMY_DATABASE_URL = "sqlite:///./uploaded_files.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
