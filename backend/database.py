from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL configuration with automatic cloud fallback
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Try MySQL if locally defined, otherwise default to SQLite for cloud/Render deployment
    DATABASE_URL = os.getenv("MYSQL_URL", "mysql+pymysql://root:%23Yash01.@localhost:3306/dr_medical_ai")

# Fix postgres:// -> postgresql:// for Render PostgreSQL if provided
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

try:
    if "sqlite" in DATABASE_URL:
        engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
    else:
        engine = create_engine(DATABASE_URL, echo=False)
    # Quick probe to check if MySQL/Postgres is reachable
    with engine.connect() as conn:
        pass
except Exception as e:
    print(f"[DB Notice] Could not connect to external DB ({e}). Falling back to local SQLite database.")
    DATABASE_URL = "sqlite:///./medical_ai.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
