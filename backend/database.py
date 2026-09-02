from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL from environment or default to local MySQL / SQLite fallback
DEFAULT_DB_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:%23Yash01.@localhost:3306/dr_medical_ai")
if DEFAULT_DB_URL:
    DEFAULT_DB_URL = DEFAULT_DB_URL.strip().strip('"').strip("'")

try:
    if DEFAULT_DB_URL.startswith("sqlite"):
        engine = create_engine(DEFAULT_DB_URL, connect_args={"check_same_thread": False}, echo=False)
    else:
        engine = create_engine(DEFAULT_DB_URL, echo=False)
    # Test connection
    with engine.connect() as conn:
        pass
except Exception as e:
    print(f"[DB WARNING] Could not connect using '{DEFAULT_DB_URL}' ({e}). Falling back to SQLite database.")
    SQLITE_URL = "sqlite:///./dr_portal.db"
    engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False}, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
