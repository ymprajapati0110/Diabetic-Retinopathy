from database import SessionLocal
import bcrypt
from models import User
from auth_service import get_password_hash

def create_verified_user():
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == "rajveerrai2807@gmail.com").first()
        if existing:
            existing.role = "verified"
            existing.hashed_password = get_password_hash("123")
            db.commit()
            print("Updated existing user to verified. Password is '123'")
        else:
            user = User(
                email="rajveerrai2807@gmail.com",
                name="Dr. Rajveer Rai",
                medical_license="MD-1234",
                hashed_password=get_password_hash("123"),
                role="verified"
            )
            db.add(user)
            db.commit()
            print("Created new verified user. Password is '123'")
    finally:
        db.close()

if __name__ == "__main__":
    create_verified_user()
