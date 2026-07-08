import asyncio
from motor.motor_asyncio import AsyncIOMotorClient

async def delete_test_user():
    client = AsyncIOMotorClient('mongodb://localhost:27017')
    db = client["dr_medical_ai"]
    
    result = await db.users.delete_many(
        {"email": "rajveerrai2807@gmail.com"}
    )
    print(f"Deleted {result.deleted_count} users")
    client.close()

if __name__ == "__main__":
    asyncio.run(delete_test_user())
