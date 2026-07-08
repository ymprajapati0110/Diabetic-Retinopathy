from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# User's specified MySQL connection
# Format: mysql+pymysql://user:password@host:port/dbname
MYSQL_URL = os.getenv("DATABASE_URL", "mysql+pymysql://root:%23Yash01.@localhost:3306/dr_medical_ai")

engine = create_engine(MYSQL_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
