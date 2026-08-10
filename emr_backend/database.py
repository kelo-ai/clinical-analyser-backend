"""
EMR System - separate database from the AI Clinical Assistant.
This is a deliberately independent service with its own storage, reflecting
that the EMR is the authoritative, long-term patient record - not just a
feature of the AI module.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./emr_system.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
