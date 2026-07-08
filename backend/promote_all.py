from database import SessionLocal
from models import User

def main():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        print("=== ALL USERS ===")
        for u in users:
            print(f"  Email: {u.email} | Role: {u.role}")
        
        # Verify ALL pending users
        pending_users = db.query(User).filter(User.role == "pending").all()
        for u in pending_users:
            u.role = "verified"
        db.commit()
        
        print(f"\n=== VERIFIED {len(pending_users)} PENDING ACCOUNT(S) ===")
        
        # Confirm
        users = db.query(User).all()
        print("\n=== UPDATED USERS ===")
        for u in users:
            print(f"  Email: {u.email} | Role: {u.role}")
    finally:
        db.close()

if __name__ == "__main__":
    main()
