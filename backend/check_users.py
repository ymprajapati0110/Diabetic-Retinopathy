from database import SessionLocal
from models import User

def check():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        for u in users:
            print(f"ID: {u.id} | Email: {u.email} | Name: {u.name} | Role: {u.role} | PwdHash: {u.hashed_password}")
    finally:
        db.close()

if __name__ == "__main__":
    check()
