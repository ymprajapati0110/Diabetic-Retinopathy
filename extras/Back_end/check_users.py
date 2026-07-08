import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def check():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client["dr_medical_ai"]
    users = await db.users.find().to_list(100)
    print("Users in DB:", users)
    client.close()

if __name__ == "__main__":
    asyncio.run(check())
