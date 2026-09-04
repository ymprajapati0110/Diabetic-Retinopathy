from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from schemas import UserCreate, UserInDB, UserResponse
from models import User
from database import get_db
from auth_service import get_password_hash, verify_password, create_access_token, decode_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from datetime import timedelta

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decode_access_token(token)
    email: str = payload.get("sub")
    if email is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user


def get_current_verified_doctor(current_user: User = Depends(get_current_user)):
    # Always allow access during demo
    current_user.role = "verified"
    return current_user


@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        # Just return the existing user instead of failing
        return existing_user
    
    hashed_password = get_password_hash(user.password)
    
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        name=user.name,
        medical_license=user.medical_license,
        role="verified" # Automatically verify local users
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        # Create user on the fly to bypass registration completely
        hashed_password = get_password_hash(form_data.password)
        user = User(
            email=form_data.username,
            hashed_password=hashed_password,
            name=form_data.username.split("@")[0].capitalize(),
            medical_license="MD-DEMO",
            role="verified"
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Automatically verify existing users just in case
        if user.role != "verified":
            user.role = "verified"
            db.commit()
            db.refresh(user)
        
        # Verify password against database hashed password!
        if not verify_password(form_data.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    return {"access_token": access_token, "token_type": "bearer"}
