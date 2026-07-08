import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import sys

async def verify_doctor(email: str):
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client["dr_medical_ai"]
    
    result = await db.users.update_one(
        {"email": email},
        {"$set": {"role": "verified"}}
    )
    
    if result.matched_count:
        print(f"Successfully verified doctor with email: {email}")
    else:
        print(f"Error: Could not find user with email: {email}")
    
    client.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_doctor.py <doctor_email>")
        sys.exit(1)
        
    asyncio.run(verify_doctor(sys.argv[1]))
