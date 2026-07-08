import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import bcrypt

async def create_verified_user():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client["dr_medical_ai"]
    
    # Check if exists
    existing = await db.users.find_one({"email": "rajveerrai2807@gmail.com"})
    if existing:
        await db.users.update_one(
            {"email": "rajveerrai2807@gmail.com"},
            {"$set": {"role": "verified", "hashed_password": bcrypt.hashpw(b"123", bcrypt.gensalt()).decode('utf-8')}}
        )
        print("Updated existing user to verified. Password is '123'")
    else:
        user_data = {
            "email": "rajveerrai2807@gmail.com",
            "name": "Dr. Rajveer Rai",
            "medical_license": "MD-1234",
            "hashed_password": bcrypt.hashpw(b"123", bcrypt.gensalt()).decode('utf-8'),
            "role": "verified"
        }
        await db.users.insert_one(user_data)
        print("Created new verified user. Password is '123'")

    client.close()

if __name__ == "__main__":
    asyncio.run(create_verified_user())
